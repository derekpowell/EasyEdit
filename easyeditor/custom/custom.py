from ..editors import BaseEditor
from ..models.rome import ROMEHyperParams
from ..util import nethook

import transformers
import torch
import torch.nn.functional as F
from transformers import GPTJForCausalLM, AutoTokenizer, AutoModel, GPT2LMHeadModel, AutoModelForCausalLM
import pandas as pd

from contextlib import redirect_stdout

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pad_token(token):
    token = " " + token if token[0] != " " else token
    return(token)


def encode_token(token:str, tokenizer, pad = True):        
    token = pad_token(token) if pad else token
    token_id = tokenizer(token)["input_ids"]
    
    # deal with sentencepiece tokenizer
    if type(tokenizer) == transformers.models.llama.tokenization_llama.LlamaTokenizer:
        return token_id[1:]

    return(token_id)


class EditedModel:
    def __init__(self, hparams):
        self.editor = BaseEditor.from_hparams(hparams)

        self.model = self.editor.model
        self.tok = self.editor.tok
        self.model_name = self.editor.model_name
        

        self.params = hparams
        self.preprompt = ""
        self.saved_weights = None
        
        self.tok.padding_side = "left"
        # self.tok.pad_token = self.tok.eos_token
    
    def edit(self, rewrite, **kwargs):
        
        if "preprompt" in rewrite: # this is a little hacky
            self.preprompt = rewrite["preprompt"]
            return None
        else:
            with redirect_stdout(None):
                metrics, self.model, self.saved_weights = self.editor.pure_edit(
                    **rewrite,
                    # **kwargs,
                    keep_original_weight = True,
                    verbose = False
                )

        return metrics
    
    
    def restore(self):

        self.preprompt = ""
        
        if self.saved_weights:
            try:
                with torch.no_grad():
                    for k, v in self.saved_weights.items():
                        nethook.get_parameter(self.model, k)[...] = v
                self.saved_weights = None
                # print("Original model restored")
            except NameError as e:
                print(f"No model weights to restore: {e}")

            
    def generate_text(self, texts, **kwargs):
        
        if type(texts) != list:
            texts = [texts]
        
        texts = [self.preprompt + t for t in texts]

        model = self.model
        tokenizer = self.tok
        encoding = tokenizer(texts, padding=True, return_tensors='pt').to(device)

        with torch.no_grad():
            generated_ids = model.generate(**encoding, **kwargs) # 

            generated_texts = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            
        return(generated_texts)
    
    
    def logprobs(self, texts):
        
        texts = self.preprompt + texts if type(texts)==str else [self.preprompt + t for t in texts]
    
        tokenizer = self.tok 
        model = self.model
        encoding = tokenizer(texts, padding=True, return_tensors='pt').to(device)

        with torch.no_grad():
            model_out = model(encoding["input_ids"])
            logits = model_out.logits
            logprobs = F.log_softmax(logits, -1)
        
        return {"tokens": encoding, "logprobs": logprobs}

    
    def completion_logprob(self, text, completion, start_ind = None):
        
        '''
        Compute model log probability of completion substring. Returns single value tensor. Takes only one text string.
        '''
        
        # texts = self.preprompt + text
    
        # tokenizer = self.tok 
        # model = self.model
        # encoding = tokenizer(texts, padding=True, return_tensors='pt').to(device)

        # with torch.no_grad():
        #     model_out = model(encoding["input_ids"])
        #     logits = model_out.logits
        #     logprobs = F.log_softmax(logits, -1)

        # token_id = encode_token(completion, tokenizer)
        # start_ind = -len(token_id)-1 if not start_ind else start_ind
        
        # l = logprobs[:, start_ind:-1, token_id]
        # if len(l.squeeze().shape) == 0:
        #     return(l.squeeze())
        # else:
        #     return(l.squeeze().diag().sum())
        

        return self.substring_logprobs(text, completion)[0][-1]
        

    def substring_logprobs(self, texts, substring, pad = True):
        '''
        Compute model log probability of each occurrence of substring in text. Returns list of list-type. Accepts a list of strings.
        '''
        
        if type(texts) != list:
            texts = [texts]
        
        logprobs = self.logprobs(texts)
        
        tok_encoded = encode_token(substring, self.tok, pad = pad)
        # text_encoded = logprobs['tokens']['input_ids'][0].tolist()
        
        out = []
        for i in range(len(texts)):
            text_encoded = logprobs['tokens']['input_ids'][i].tolist()

            # find matches for searched token sequence
            start_idxs = []
            for left in range(0, len(text_encoded) - len(tok_encoded)+1):
                # left = i - 1
                right = left + len(tok_encoded)
                if text_encoded[left:right] == tok_encoded:
                    start_idxs.append(left)

            lp = logprobs['logprobs'][i]
            match_probs = []

            # compute probability for all tokens
            for start in start_idxs:
                val = 0
                for i in range(len(tok_encoded)):
                    val += lp[start + i - 1][tok_encoded[i]]
                match_probs.append(val)

            out.append(match_probs)

        return out
        

    def choose(self, prompt, choices, normalization = None):

        padded_choices = [pad_token(c) for c in choices]
        prompts = [prompt + c for c in padded_choices]

        logits = torch.tensor([self.completion_logprob(prompts[i], padded_choices[i]) for i in range(len(padded_choices))])

        if normalization == "unconditional":
            norm_logits = torch.tensor([self.completion_logprob(padded_choices[i], padded_choices[i]) for i in range(len(padded_choices))])
            logits = logits - norm_logits

        elif normalization == "byte_length":    
            str_lens = [len(c) for c in choices]
            logits = logits / torch.tensor(str_lens)

        elif normalization == "token_length":
            tok_lens = [len(encode_token(c, self.tok)) for c in choices]
            logits = logits / torch.tensor(tok_lens)

        elif normalization == "root":
            tok_lens = [len(encode_token(c, self.tok)) for c in choices]
            logits = torch.pow(torch.exp(logits), 1./torch.tensor(tok_lens))

        logits = logits.tolist()

        return(logits.index(max(logits)))
    

def evaluate(evaluation_data, model, prefix_fwd = "", prefix_rev = ""):

    fwd_answers = []
    rev_answers = []
    corr_fwd_answers = []
    corr_rev_answers = []

    for q in evaluation_data.itertuples():

        fwd_choices =  q.fwd_choices
        query_fwd = q.query_fwd.replace("<subj>", q.subj).replace("<answer>", "")
        query_fwd = prefix_fwd + query_fwd
        ans_fwd = model.choose(query_fwd, fwd_choices, normalization = None) # None, "unconditional", "byte_length", "token_length", "root"
        corr_fwd_answers.append(fwd_choices.index(q.answer_fwd))
        fwd_answers.append(ans_fwd)

        rev_choices =  q.rev_choices
        query_rev = q.query_rev.replace("<answer>", q.answer_fwd).replace("<subj>", "")
        query_rev = prefix_rev + query_rev
        ans_rev = model.choose(query_rev, rev_choices, normalization = None) # None, "unconditional", "byte_length", "token_length", "root"
        corr_rev_answers.append(rev_choices.index(q.subj))
        rev_answers.append(ans_rev)

    results = (
        evaluation_data
        .assign(
            corr_fwd_answer = corr_fwd_answers,
            corr_rev_answer = corr_rev_answers,
            fwd_predicted = fwd_answers,
            rev_predicted = rev_answers
            )
        .assign(
            correct_fwd = lambda x: x.corr_fwd_answer==x.fwd_predicted,
            correct_rev = lambda x: x.corr_rev_answer==x.rev_predicted
        )
    )

    return(results)


def edit_and_evaluate(edits_df, eval_df, model, edit_method):
    
    full_results = pd.DataFrame()

    for e in edits_df.itertuples():
        if edit_method == "ROME":
            rewrite = {
                'prompts': [f'A {e.subj} is a'],
                'target_new': [e.entity], #{'str': e.entity},
                'subject': [e.subj]
            }
            edited_model.edit(rewrite)
            
        elif edit_method == "ICE":
            edited_model.edit({"preprompt": f"Imagine a {e.subj} was a kind of {e.entity}. "}) # and not a kind of {e.orig_entity}

        evals = eval_df.loc[lambda x: x.entity == e.entity]
        res = evaluate(evals, edited_model)
        
        edited_model.restore()

        full_results = pd.concat([full_results, res])

    full_results["edit_method"] = edit_method

    return(full_results)