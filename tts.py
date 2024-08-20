import ChatTTS
import torch
import torchaudio

chat = ChatTTS.Chat()
chat.load(compile=True) # Set to True for better performance

texts = ["""This man was discovered to be unknowingly missing 90% of his brain, yet he was living a normal life, How is this even possible?

Meet Nade, a living example of extreme brain damage. Despite having only 10% of his brain functioning, Nade has mastered complex skills like playing the piano and solving intricate mathematical problems!

Thanks to the incredible plasticity of the brain, Nade has adapted in ways you wouldn’t believe. Remarkably, he can perform everyday tasks and even excel in certain areas, demonstrating the brain's incredible ability to rewire and compensate.

Think you know someone who’s got 10% of their brain left? Tag them below! And drop a comment with your thoughts on the brain’s amazing adaptability!

Don’t miss out on more fascinating videos like this! Follow us for more intriguing facts and stories that’ll keep you coming back for more!"""]

###################################
# Sample a speaker from Gaussian.

rand_spk = chat.sample_random_speaker()
print(rand_spk) # save it for later timbre recovery

params_infer_code = ChatTTS.Chat.InferCodeParams(
    spk_emb = rand_spk, # add sampled speaker 
    temperature = .3,   # using custom temperature
    top_P = 0.7,        # top P decode
    top_K = 20,         # top K decode
)

###################################
# For sentence level manual control.

# use oral_(0-9), laugh_(0-2), break_(0-7) 
# to generate special token in text to synthesize.
params_refine_text = ChatTTS.Chat.RefineTextParams(
    prompt='[oral_2][laugh_0][break_6]',
)

wavs = chat.infer(
    texts,
    params_refine_text=params_refine_text,
    params_infer_code=params_infer_code,
)


for i in range(len(wavs)):
    torchaudio.save(f"basic_output{i}.wav", torch.from_numpy(wavs[i]).unsqueeze(0), 24000)
    

# ###################################
# # For word level manual control.

# text = 'What is [uv_break]your favorite english food?[laugh][lbreak]'
# wavs = chat.infer(text, skip_refine_text=True, params_refine_text=params_refine_text,  params_infer_code=params_infer_code)
# torchaudio.save("word_level_output.wav", torch.from_numpy(wavs[0]).unsqueeze(0), 24000)    