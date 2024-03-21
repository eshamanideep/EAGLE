import os
import time
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import gradio as gr
import argparse
from model.ea_model import EaModel
import torch
from fastchat.model import get_conversation_template
import re


torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future
torch._dynamo.config.cache_size_limit=64


def truncate_list(lst, num):
    if num not in lst:
        return lst


    first_index = lst.index(num)


    return lst[:first_index + 1]





def find_list_markers(text):

    pattern = re.compile(r'(?m)(^\d+\.\s|\n)')
    matches = pattern.finditer(text)


    return [(match.start(), match.end()) for match in matches]


def checkin(pointer,start,marker):
    for b,e in marker:
        if b<=pointer<e:
            return True
        if b<=start<e:
            return True
    return False

def highlight_text(text, text_list,color="black"):

    pointer = 0
    result = ""
    markers=find_list_markers(text)


    for sub_text in text_list:

        start = text.find(sub_text, pointer)
        if start==-1:
            continue
        end = start + len(sub_text)


        if checkin(pointer,start,markers):
            result += text[pointer:start]
        else:
            result += f"<span style='color: {color};'>{text[pointer:start]}</span>"

        result += sub_text

        pointer = end

    if pointer < len(text):
        result += f"<span style='color: {color};'>{text[pointer:]}</span>"

    return result


def warmup(model):
    conv = get_conversation_template(args.model_type)

    if args.model_type == "llama-2-chat":
        sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
        conv.system_message = sys_p
    conv.append_message(conv.roles[0], "Hello")
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    if args.model_type == "llama-2-chat":
        prompt += " "
    input_ids = model.tokenizer([prompt]).input_ids
    input_ids = torch.as_tensor(input_ids).cuda()
    for output_ids in model.chainea_generate(input_ids):
        ol=output_ids.shape[1]


def warmupbase(model):
    conv = get_conversation_template(args.model_type)

    if args.model_type == "llama-2-chat":
        sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
        conv.system_message = sys_p
    conv.append_message(conv.roles[0], "Hello")
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    if args.model_type == "llama-2-chat":
        prompt += " "
    input_ids = model.tokenizer([prompt]).input_ids
    input_ids = torch.as_tensor(input_ids).cuda()
    for output_ids in model.naive_generate(input_ids):
        ol=output_ids.shape[1]

def bot(history, session_state,):
    temperature, top_p, use_EaInfer, highlight_EaInfer =0.0,0.0,False,True
    if not history:
        return history, "0.00 tokens/s", "0.00", session_state
    pure_history = session_state.get("pure_history", [])
    assert args.model_type == "llama-2-chat" or "vicuna"
    conv = get_conversation_template(args.model_type)

    if args.model_type == "llama-2-chat":
        sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
        conv.system_message = sys_p

    for query, response in pure_history:
        conv.append_message(conv.roles[0], query)
        if args.model_type == "llama-2-chat" and response:
            response = " " + response
        conv.append_message(conv.roles[1], response)

    prompt = conv.get_prompt()

    if args.model_type == "llama-2-chat":
        prompt += " "

    input_ids = model.tokenizer([prompt]).input_ids
    input_ids = torch.as_tensor(input_ids).cuda()
    input_len = input_ids.shape[1]
    naive_text = []
    cu_len = input_len
    totaltime=0
    torch.cuda.synchronize()
    start_time=time.time()
    total_ids=0
    if use_EaInfer:

        for output_ids in model.chainea_generate(input_ids, temperature=temperature, top_p=top_p,
                                            max_new_tokens=args.max_new_token):
            torch.cuda.synchronize()
            totaltime+=(time.time()-start_time)
            total_ids+=1
            decode_ids = output_ids[0, input_len:].tolist()
            decode_ids = truncate_list(decode_ids, model.tokenizer.eos_token_id)
            text = model.tokenizer.decode(decode_ids, skip_special_tokens=True, spaces_between_special_tokens=False,
                                          clean_up_tokenization_spaces=True, )
            naive_text.append(model.tokenizer.decode(output_ids[0, cu_len], skip_special_tokens=True,
                                                     spaces_between_special_tokens=False,
                                                     clean_up_tokenization_spaces=True, ))


            #print(output_ids)

            cu_len = output_ids.shape[1]
            colored_text = highlight_text(text, naive_text, "orange")
            if highlight_EaInfer:
                history[-1][1] = colored_text
            else:
                history[-1][1] = text
            pure_history[-1][1] = text
            session_state["pure_history"] = pure_history
            new_tokens = cu_len-input_len
            # print(new_tokens)
            # print(totaltime)
            yield history,f"{new_tokens/totaltime:.2f} tokens/s",f"{new_tokens/total_ids:.2f}",session_state
            torch.cuda.synchronize()
            start_time = time.time()


    else:
        for output_ids in model.naive_generate(input_ids, temperature=temperature, top_p=top_p,
                                               max_new_tokens=args.max_new_token):
            torch.cuda.synchronize()
            totaltime += (time.time() - start_time)
            total_ids+=1
            decode_ids = output_ids[0, input_len:].tolist()
            decode_ids = truncate_list(decode_ids, model.tokenizer.eos_token_id)
            text = model.tokenizer.decode(decode_ids, skip_special_tokens=True, spaces_between_special_tokens=False,
                                          clean_up_tokenization_spaces=True, )
            naive_text.append(model.tokenizer.decode(output_ids[0, cu_len], skip_special_tokens=True,
                                                     spaces_between_special_tokens=False,
                                                     clean_up_tokenization_spaces=True, ))
            cu_len = output_ids.shape[1]
            colored_text = highlight_text(text, naive_text, "orange")
            if highlight_EaInfer and use_EaInfer:
                history[-1][1] = colored_text
            else:
                history[-1][1] = text
            history[-1][1] = text
            pure_history[-1][1] = text
            new_tokens = cu_len - input_len
            # print(new_tokens)
            # print(totaltime)
            yield history,f"{new_tokens/totaltime:.2f} tokens/s",f"{new_tokens/total_ids:.2f}",session_state
            torch.cuda.synchronize()
            start_time = time.time()


def user(user_message, history,session_state):
    if history==None:
        history=[]
    pure_history = session_state.get("pure_history", [])
    pure_history += [[user_message, None]]
    session_state["pure_history"] = pure_history
    return "", history + [[user_message, None]],session_state


def regenerate(history,session_state):
    if not history:
        return history, None,"0.00 tokens/s","0.00",session_state
    pure_history = session_state.get("pure_history", [])
    pure_history[-1][-1] = None
    session_state["pure_history"]=pure_history
    if len(history) > 1:  # Check if there's more than one entry in history (i.e., at least one bot response)
        new_history = history[:-1]  # Remove the last bot response
        last_user_message = history[-1][0]  # Get the last user message
        return new_history + [[last_user_message, None]], None,"0.00 tokens/s","0.00",session_state
    history[-1][1] = None
    return history, None,"0.00 tokens/s","0.00",session_state


def clear(history,session_state):
    pure_history = session_state.get("pure_history", [])
    pure_history = []
    session_state["pure_history"] = pure_history
    return [],"0.00 tokens/s","0.00",session_state




parser = argparse.ArgumentParser()
parser.add_argument(
    "--ea-model-path",
    type=str,
    default="/home/lyh/weights/hf/eagle/llama2chat/13B/model_int4.g32.pth",
    help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
)
parser.add_argument("--base-model-path", type=str, default="/home/lyh/weights/hf/llama2chat/13B/model_int4.g32.pth",
                    help="path of basemodel, huggingface project or local path")
parser.add_argument(
    "--load-in-8bit", action="store_true", help="Use 8-bit quantization"
)
parser.add_argument(
    "--load-in-4bit", action="store_true", help="Use 4-bit quantization"
)
parser.add_argument("--model-type", type=str, default="llama-2-chat", help="llama-2-chat or vicuna, for chat template")
parser.add_argument(
    "--max-new-token",
    type=int,
    default=1024,
    help="The maximum number of new generated tokens.",
)
args = parser.parse_args()

model = EaModel.from_pretrained(
    base_model_path=args.base_model_path,
    ea_model_path=args.ea_model_path,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    load_in_4bit=args.load_in_4bit,
    load_in_8bit=args.load_in_8bit,
    device_map="auto"
)
model.eval()

print("Compiling....")
model.draft_one=torch.compile(model.draft_one, mode="reduce-overhead", fullgraph=True,dynamic=False)
#model.draft_many=torch.compile(model.draft_many, mode="reduce-overhead", fullgraph=True)
model.base_forward=torch.compile(model.base_forward, mode="reduce-overhead", fullgraph=True,dynamic=False)
model.base_forward_one=torch.compile(model.base_forward_one, mode="reduce-overhead", fullgraph=True)
print("Compilation done")

print("Warming up")
warmupbase(model)
warmupbase(model)
print("Warmup done!")

custom_css = """
#speed textarea {
    color: red;   
    font-size: 30px; 
}"""

with gr.Blocks(css=custom_css) as demo:
    gs = gr.State({"pure_history": []})
    #gr.Markdown('''## EAGLE Chatbot''')
    with gr.Row():
        speed_box = gr.Textbox(label="Speed", elem_id="speed", interactive=False, value="0.00 tokens/s")
        compression_box = gr.Textbox(label="Compression Ratio", elem_id="speed", interactive=False, value="0.00")
    # with gr.Row():
    #     with gr.Column():
    #         use_EaInfer = gr.Checkbox(label="Use EAGLE", value=True)
    #         highlight_EaInfer = gr.Checkbox(label="Highlight the tokens generated by EAGLE", value=True)
    #     temperature = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="temperature", value=0.5)
    #     top_p = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="top_p", value=0.9)
    # note=gr.Markdown(show_label=False,interactive=False,value='''The Compression Ratio is defined as the number of generated tokens divided by the number of forward passes in the original LLM. If "Highlight the tokens generated by EAGLE" is checked, the tokens correctly guessed by EAGLE
    # will be displayed in orange. Note: Checking this option may cause special formatting rendering issues in a few cases, especially when generating code''')

    # note = gr.Markdown(show_label=False, value='''The Compression Ratio is defined as the number of generated tokens divided by the number of forward passes in the original LLM. If "Highlight the tokens generated by EAGLE" is checked, the tokens correctly guessed by EAGLE
    #     will be displayed in orange. Note: Checking this option may cause special formatting rendering issues in a few cases, especially when generating code''')


    chatbot = gr.Chatbot(height=800,show_label=False)


    msg = gr.Textbox(label="Your input")
    with gr.Row():
        send_button = gr.Button("Send")
        stop_button = gr.Button("Stop")
        regenerate_button = gr.Button("Regenerate")
        clear_button = gr.Button("Clear")
    enter_event=msg.submit(user, [msg, chatbot,gs], [msg, chatbot,gs], queue=True).then(
        bot, [chatbot, gs], [chatbot,speed_box,compression_box,gs]
    )
    clear_button.click(clear, [chatbot,gs], [chatbot,speed_box,compression_box,gs], queue=True)

    send_event=send_button.click(user, [msg, chatbot,gs], [msg, chatbot,gs],queue=True).then(
        bot, [chatbot, gs], [chatbot,speed_box,compression_box,gs]
    )
    regenerate_event=regenerate_button.click(regenerate, [chatbot,gs], [chatbot, msg,speed_box,compression_box,gs],queue=True).then(
        bot, [chatbot, gs], [chatbot,speed_box,compression_box,gs]
    )
    stop_button.click(fn=None, inputs=None, outputs=None, cancels=[send_event,regenerate_event,enter_event])
demo.queue()
demo.launch(share=True)
