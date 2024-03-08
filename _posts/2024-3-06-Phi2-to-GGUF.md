---
title:  "Scaling Down, Boosting Up: Converting Microsoft Phi-2 to GGUF format for Compact Deployments"
layout: post
excerpt_separator: <!-- excerpt -->
---

![Robotic Russian Dolls](/assets/images/gguf-russian_dolls.jpeg){:style="display:block; margin-left:auto; margin-right:auto"}

In my [previous blog post](https://medium.com/thedeephub/optimizing-phi-2-a-deep-dive-into-fine-tuning-small-language-models-9d545ac90a99), I discussed how we can finely tune a small language model, specifically Microsoft Phi-2, and effectively train it on a single GPU to achieve remarkable results. Following that post, a comment caught my attention, where a reader inquired about generating a GGUF file from the fine-tuned model. While I had some knowledge on the topic, I wanted to explore this a bit more. With a little bit of research on the web, I realized that there was not a lot of content on this topic and figured that it might be worthwhile to post a more detailed article on this.  

For those who are uninitiated, GGUF is a successor to GGML (GPT-Generated Model Language) developed by the [brilliant Georgi Gerganov and the llama.cpp team](https://github.com/ggerganov/llama.cpp). It allows for faster and more efficient use of language models for tasks like text generation, translation, and question answering. GGUF is quite popular among macOS users and due to its minimal setup and performance, you can run inference on pretty much any operating system, including running it on a Docker container. How cool is that!  

<!-- excerpt -->  

I am using Google Colab for my code, and I will be using llama.cpp to convert the fine-tuned model to GGUF and to spice things up I am using LangChain with llama-cpp-python, which is a python binding for llama.cpp for running inference. I am using a GPU instance to convert the model to GGUF format and then I will switch to a CPU instance to run inference, to demonstrate that it works great on a CPU.  

Here are the links to the source code in Google Colab notebook and the GitHub repo. Feel free to reuse for your own projects. Happy Coding!  

[Google Colab Notebook](https://colab.research.google.com/drive/1JQQJE3OTv_60U24J6arRsMrXfCCJ1MMO)  

[GitHub Repository](https://github.com/yernenip/phi2-gguf/)  

# Loading and Merging Phi-2 with fine-tuned LoRA adapters
Before we get into converting the fine-tuned model to GGUF format, lets first load the model and merge it with the LoRA adapters. I am going to reuse the adapters I created in my previous blog post. We will merge the adapters and push it to hugging face hub. Let’s start with installing the python packages we need at this stage.  

{% highlight python %}
!pip install peft
!pip install --upgrade torch transformers
{% endhighlight %}  

Next, we will load the model in fp16. Note that in my previous blog I had used bits and bytes to load it in 4-bit, reducing the memory usage. However, in this case we cannot convert models that have already been quantized (8-bit? or below) to GGUF format.  

{% highlight python %}
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "microsoft/phi-2"
torch.set_default_device("cuda")

model = AutoModelForCausalLM.from_pretrained(model_name,
                                             torch_dtype=torch.float16,
                                             trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
{% endhighlight %}  

Once the model is downloaded (this could take a while), lets load and merge the adapters to the model.  

{% highlight python %}
from peft import PeftModel, PeftConfig

#Load the model weights from hub
model_adapters = "praveeny/phi2-webglm-qlora"
model = PeftModel.from_pretrained(model, model_adapters)

model = model.merge_and_unload()
model.save_pretrained("updated_adapters")
{% endhighlight %}  

After the model has been merged and saved to local, you can see the files by clicking on the “Files” icon on the left in Google Colab, we will push the model to hugging face hub.  

{% highlight python %}
model.push_to_hub("phi2-webglm-guava", private=True,
                  commit_message="merged model")

tokenizer.push_to_hub("phi2-webglm-guava", private=True,
                  commit_message="tokenizer")
{% endhighlight %}  

We will also need to push the tokenizer along with the model as this would be needed for conversion later down the line.  

# Installing and building llama.cpp and converting HF model to GGUF
At this stage, if you are using Google Colab, I would recommend to disconnect and delete runtime. Let’s download the merged model, install and build llama.cpp and convert the downloaded model to GGUF.  

Below is the code to download the model along with the tokenizer.  

{% highlight python %}
from huggingface_hub import snapshot_download

#Download the merged model
model_id="praveeny/phi2-webglm-guava"
#Download the repository to local_dir
snapshot_download(repo_id=model_id, local_dir="phi2",
                  local_dir_use_symlinks=False)
{% endhighlight %}  

Installing and building llama.cpp  

{% highlight python %}
#Setup Llama.cpp and install required packages
!git clone https://github.com/ggerganov/llama.cpp
!cd llama.cpp && LLAMA_CUBLAS=1 make
!pip install -r llama.cpp/requirements.txt
{% endhighlight %}  

I am using CuBLAS, which is a library provided by NVIDIA for performing BLAS operations on NVIDIA GPUs. It is optimized for the NVIDIA CUDA architecture. BLAS (Basic Linear Algebra Subprograms) is a library that provides optimized routines for performing common linear algebra operations. By utilizing BLAS, llama.cpp can perform mathematical calculations more efficiently, leading to faster processing of the input text. Llama.cpp offers several different BLAS implementations for usage.  

We then run “make”, a software tool that automates the process of building software. Its typically used for compiling C and C++ projects.  

Now that we have the project compiled and the necessary python packages installed, lets run the convert on the downloaded model.  

{% highlight python %}
!python llama.cpp/convert-hf-to-gguf.py phi2 --outfile "phi2/phi2-v2-fp16.bin" --outtype f16
{% endhighlight %}  

Note that because this is a hugging face model we downloaded, I am using the “convert-hf-to-gguf.py” instead of the “convert.py”. We save the output bin file in the phi2 directory.

Our next step would be to quantize the converted model to Q5_K_M.  

{% highlight python %}
!./llama.cpp/quantize "phi2/phi2-v2-fp16.bin" "phi2/phi2-v2-Q5_K_M.gguf" "q5_k_m"
{% endhighlight %}  

Notice the “q5_k_m” in the command. Q5 refers to the number of bits used to represent the quantized weights, so the weights are represented using 5 bits. “k” refers to the key weights within the attention mechanism. I could not find reference to “m” in any public documentation, but if I were to guess it probably means moderate level of quantization? I used this setting as it preserves the model performance, while also being optimal for memory usage. Lower bits can decrease performance, while higher bits, can take up more memory.  

After the model has been quantized, lets upload the GGUF to a different repo on hugging face.  

{% highlight python %}
!pip install huggingface_hub

from huggingface_hub import HfApi
api = HfApi()

model_id = "praveeny/phi2-webglm-gguf"
api.create_repo(model_id, exist_ok=True, repo_type="model")
api.upload_file(
    path_or_fileobj="phi2/phi2-v2-Q5_K_M.gguf",
    path_in_repo="phi2-v2-Q5_K_M.gguf",
    repo_id=model_id,
)
{% endhighlight %}  

# Running Inference with LangChain, llama.cpp and GGUF format
Like before, if you are on Google Colab, I would recommend to disconnect and delete your runtime. And go ahead and connect to a CPU instance. First we will install the necessary packages to run inference, then download the GGUF from Hugging Face and finally run inference on it.  

{% highlight python %}
!pip install huggingface_hub
!pip install langchain
!pip install llama-cpp-python
{% endhighlight %}  

Downloading the GGUF file from hugging face  

{% highlight python %}
from huggingface_hub import snapshot_download

model_id="praveeny/phi2-webglm-gguf"
#Download the repository to local_dir
snapshot_download(repo_id=model_id, local_dir="phi2-gguf",
                  local_dir_use_symlinks=False)
{% endhighlight %}  

Setting up LangChain, prompt and running inference  

{% highlight python %}
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="phi2-gguf/phi2-v2-Q5_K_M.gguf",
    temperature=0.75,
    max_tokens=2000,
    top_p=1,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)


prompt = """###System:
Read the references provided and answer the corresponding question.
###References:
[1] For most people, the act of reading is a reward in itself. However, studies show that reading books also has benefits that range from a longer life to career success. If you’re looking for reasons to pick up a book, read on for seven science-backed reasons why reading is good for your health, relationships and happiness.
[2] As per a study, one of the prime benefits of reading books is slowing down mental disorders such as Alzheimer’s and Dementia  It happens since reading stimulates the brain and keeps it active, which allows it to retain its power and capacity.
[3] Another one of the benefits of reading books is that they can improve our ability to empathize with others. And empathy has many benefits – it can reduce stress, improve our relationships, and inform our moral compasses.
[4] Here are 10 benefits of reading that illustrate the importance of reading books. When you read every day you:
[5] Why is reading good for you? Reading is good for you because it improves your focus, memory, empathy, and communication skills. It can reduce stress, improve your mental health, and help you live longer. Reading also allows you to learn new things to help you succeed in your work and relationships.
###Question:
Why is reading books widely considered to be beneficial?
###Answer:
"""


llm.invoke(prompt)
{% endhighlight %}  

For the prompt, I am using the same prompt from my previous article on fine tuning Microsoft’s Phi-2.  

And below is the output!  


{% highlight html %}
Reading books is widely considered to be beneficial because it can improve 
focus, memory, empathy, and communication skills[5], reduce stress, 
improve mental health, and help you live longer[5], and allow you to 
learn new things to help you succeed in your work and relationships[5]. 
It can also slow down mental disorders such as Alzheimer’s and Dementia by 
stimulating the brain and keeping it active[2], and improve our ability to 
empathize with others which can reduce stress, improve our relationships, 
and inform our moral compasses[3]. Additionally, it can improve our ability 
to comprehend and retain information, which can help us succeed 
academically[4]. Finally, it can be rewarding in itself as it can be an 
enjoyable activity[1].
{% endhighlight %}  

On a CPU, it took 138518 milliseconds or approximately over 2 minutes, for it to run. And the GGUF file itself was around 2GB, roughly 60% less in size than the fine-tuned model I downloaded (5GB).  

# Final Thoughts


![Chat with AI](/assets/images/gguf-AIChat.jpeg){:style="display:block; margin-left:auto; margin-right:auto"}

Apart from the fact that GGUF format is compatible with CPU, and can run on Apple hardware, it offers a variety of benefits as listed below. But running inference on a desktop CPU will still be slowish, unless you are experimenting and want to work with an LLM on your computer.  

1. **Reduced Model Size:** While not the primary focus, GGUF has a smaller file size compared to the original format, depending on the model and the chosen quantization method used during conversion.
2. **Deployment on devices with limited storage:** If you plan to deploy your LLM on mobile devices or embedded systems with limited storage space, the smaller GGUF file size can be a significant advantage.
3. **Efficient Memory Usage:** GGUF is specifically designed for efficient memory usage during inference. This allows the model to run effectively on devices with less memory, expanding deployment possibilities beyond GPUs with high computational power.  

And that’s all folks! Thank you for taking time to read this article.  

# Resources

https://github.com/ggerganov/llama.cpp  

https://towardsdatascience.com/quantize-llama-models-with-ggml-and-llama-cpp-3612dfbcc172  

https://python.langchain.com/docs/integrations/llms/llamacpp  

https://huggingface.co/docs/trl/main/en/use_model  

https://huggingface.co/docs/huggingface_hub/v0.21.4/guides/download  



