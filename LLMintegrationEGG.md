In the folder Mamba4Rec exist code files mamba4rec.py, run.py, config.yaml, and environment.yaml as well as README.md. These files are cloned from the Mamba4Rec paper https://arxiv.org/abs/2403.03900, specifically their github at https://github.com/chengkai-liu/Mamba4Rec. 

I am trying to modify the Mamba4Rec code in order to integrate an LLM before prediction. Specifically, I am developing a movie recommender app which will be driven by recommendations from Mamba4Rec. However, the app will center around user interaction with an LLM chatbot where the user is able to make requests like 'I want a cozy movie from the early 2000s' or 'I want something Tarantino-style but more depressing'. The mood, inent, and sentiment of the user will be vectorized by the LLM and then these vectors need to be stored and aligned in the same latent hidden_size space that the seq_output tensor outputs of the Mamba layer live in. 

I consulted with ChatGPT and was recommended to do this using gated fusion because this method would preserve learned sequential behaviour, allow temporary mood overrides, avoid catastrophic shifts in recommendations, and learn when to trust history vs. current intent. In order to start this process I already modified the Mamba4Rec code predict() and full_sort_predict() functions to include fusion. I also created LLMprojection.py which is a minimum viable llm projection module. 

ChatGPT also suggested a training strategy that is as follows (where m_currrent is the llm vectorization of current mood and intent):

Phase 1:
Train vanilla Mamba4Rec
Freeze checkpoint
This establishes a stable item geometry

Phase 2: Add m_current only
Freeze:
item embeddings
Mamba layers
Train:
LLMprojection
fusion
Loss still BPR or CE.
Why:
Prevents the LLM from warping item space
Forces LLM vectors to align to existing geometry

Phase 3: Joint fine-tuning (optional)
Unfreeze fusion + top Mamba layer
Very low learning rate
Lets the model adapt to explicit intent

Given my ultimate goal of having a recommendation system that is driven by mamba architechture but fine-tuned and given feedback to through an LLM chat bot, taking into consideration the code in the Mamba4Rec folder, and the changes I have made so far to integrate fusion, please do the following:

1. Generate a IntegrationPlan.md in the Mamba4Rec folder where you answer the following questions:
    a. Is it best to use mamba with BPR or CE loss when integrating with an LLM?
    b. Is gated fusion the best way to integrate maba and llm for my project?
    c. Is the code set up so far able to handle fusion and if not what else is required to make it work?
    d. Is the above training plan optimal for my problem and if not suggest changes to it
    e. Describe an overall architecture for LLM integration- describe what databases would be necessary for storage before fusion, how llm vectors would be added to and aligned to the mamba output hidden_size vector space, and how new user interactions with the llm would be used to update the vectors in that space.
    f. Give a flowchart to describe how LLM interaction data (and all intermediate vectors) flows through the system (and databases) and affects predictions from mamba (how is the vector space updated with new user interactions and where does mamba model training factor into this)
2. Generate a new folder inside Mamba4Rec called Fusion and inside it replicate all the code and files in Mamba4Rec EXCEPT modify the code in any way that is necessary for fusion between the LLM and mamba4Rec to work and modify the LLMprojection.py if necesssary to make it a more sophisticated projection module. 
3. Add a section to IntegrationPlan.md where you describe in detail the code changes you made (with specific reference to the code) and what the purpose of these changes is/ how the code works.

