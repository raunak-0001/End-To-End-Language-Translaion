from transformers import AutoTokenizer
import torch
from model import Seq2SeqEncDec
import config
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
import pickle
import torch.nn.functional as F

src_sent_tokenizer=AutoTokenizer.from_pretrained("google-T5/T5-base")

if torch.cuda.is_available():
    device=torch.device("cuda")
else:
    device=torch.device("cpu")


network = Seq2SeqEncDec(config.vs,config.vd,128).to(device)

if torch.cuda.is_available():
    network.load_state_dict(torch.load("model.pth",map_location="cuda:0")) # is line me hum apne sare parameters network me daal rahe hai or map-location is liye taki sara inference logic paas kar sake
else:
    network.load_state_dict(torch.load("model.pth",map_location=torch.device("cpu")))


network.eval()

with open("dst-lang-vocab2idx.pkl","rb") as file_handle:
    vd=pickle.load(file_handle)

with open("dst-lang-idx2vocab.pkl","rb") as file_handle:
    hindi_idx2vocab=pickle.load(file_handle)

def generate_translation(eng_sentence):

    tokenized_eng_sentence = src_sent_tokenizer.tokenize(eng_sentence)
    token_ids = src_sent_tokenizer.convert_tokens_to_ids(tokenized_eng_sentence)
    token_ids_tensor = torch.tensor(token_ids)
    token_ids_tensor = torch.unsqueeze(token_ids_tensor,0)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        token_ids_tensor = token_ids_tensor.to(device)

    encoder_outputs,(final_encoder_output,final_candidate_cell_state) = network.encoder(token_ids_tensor)
    decoder_first_time_step_input = torch.tensor([[1]])

    if torch.cuda.is_available():
        encoder_outputs = encoder_outputs.to(device)
        final_encoder_output = final_encoder_output.to(device)
        final_candidate_cell_state = final_candidate_cell_state.to(device)
        decoder_first_time_step_input = decoder_first_time_step_input.to(device)

    decoder_first_time_step_output, (hidden_decoder_output, hidden_decoder_cell_state) = network.decoder(decoder_first_time_step_input,
                                                                                                        final_encoder_output,
                                                                                                        final_candidate_cell_state)

    generated_token_id = torch.argmax(F.softmax(decoder_first_time_step_output[:,0,:],dim=1),1)
    generated_token_id = torch.unsqueeze(generated_token_id,1)

    hindi_translated_sentence = str()
    hindi_translated_sentence += " " + hindi_idx2vocab[generated_token_id.item()]

    if torch.cuda.is_available():
        generated_token_id = generated_token_id.to(device)
        hidden_decoder_output = hidden_decoder_output.to(device)
        hidden_decoder_cell_state = hidden_decoder_cell_state.to(device)
        
    for i in range(config.nd-1):
        
        decoder_first_time_step_output, (hidden_decoder_output, hidden_decoder_cell_state) = network.decoder(generated_token_id,
                                                                                                hidden_decoder_output,
                                                                                                hidden_decoder_cell_state)
        generated_token_id = torch.argmax(F.softmax(decoder_first_time_step_output[:,0,:],dim=1),1)
        generated_token_id = torch.unsqueeze(generated_token_id,1)

        if torch.cuda.is_available():
            generated_token_id = generated_token_id.to(device)
            hidden_decoder_output = hidden_decoder_output.to(device)
            hidden_decoder_cell_state = hidden_decoder_cell_state.to(device)

        if generated_token_id.item() == vd["<EOS>"]:
            break

        hindi_translated_sentence += " " + hindi_idx2vocab[generated_token_id.item()]

    return hindi_translated_sentence



class TranslateRequest(BaseModel):
    text:str

app=FastAPI()

@app.post("/translate")
def translate(request: TranslateRequest):  #datatype: variable

    if not request.text:
        raise HTTPException(status_code=400,detail="The engish sentence to be translated is missing")
    try:
        hindi_translated_sentence= generate_translation(request.text)
        return {"hindi translation": hindi_translated_sentence}
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))
