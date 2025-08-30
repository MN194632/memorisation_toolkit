from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ForCTC,Wav2Vec2Processor,Wav2Vec2CTCTokenizer
import os

class BaseConfig:
    # LibriSpeech paths
    train_csv = "/work3/s194632/LibriSpeech/train-clean-100.csv"
    dev_csv = "/work3/s194632/LibriSpeech/dev-clean.csv"
    test_csv = "/work3/s194632/LibriSpeech/test-clean.csv"
    base_audio_dir = "/work3/s194632/LibriSpeech"

    model_type = 'wav2vec2'   
    # Model configuration
    tokenizer =Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base-960h")# Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-conformer-rel-pos-large")#
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")#Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-conformer-rel-pos-large")#
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h",
                                           ctc_loss_reduction="mean",
                                           pad_token_id=processor.tokenizer.pad_token_id) #Wav2Vec2ConformerForCTC.from_pretrained("facebook/wav2vec2-conformer-rel-pos-large")#
    
    # Training configuration
    batch_size = 4
    num_workers = 4
    target_sample_rate: int = 16000


######################################
####  Lowfreq Experiments PARTIAL #### 
######################################
class Config1(BaseConfig):

    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/low_freq/1x_results" #! change this for specific runs
    
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/1x_samples"
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
   
    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False  
    freeze = True
    speed = 1

class Config15(BaseConfig):

    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/low_freq/15x_results" #! change this for specific runs
    
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/15x_samples"
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
   
    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False  
    freeze = True
    speed = 1.5

class Config2(BaseConfig):
    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/low_freq/2x_results" #! change this for specific runs

    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/2x_samples"
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
   
    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False   
    freeze = True
    speed = 2

class Config25(BaseConfig):

    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/low_freq/25x_results" #! change this for specific runs
    
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/25x_samples" #!
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")

    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False   
    freeze = True
    speed = 2.5

class Config3(BaseConfig):  
    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/low_freq/3x_results" #! change this for specific runs
    
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/3x_samples" #!
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")   

    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False   
    freeze = True
    speed = 3

class Config35(BaseConfig):
    
    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/low_freq/35x_results" #! change this for specific runs
    
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/35x_samples" #!
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
    
    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False   
    freeze = True
    speed = 3.5

class Config4(BaseConfig):

    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/low_freq/4x_results" #! change this for specific runs
     
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/4x_samples" #!
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
   
    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False   
    freeze = True
    speed = 4

########################################
#### Higher Frequencies Runs PARTIAL#### 
########################################
class Config1_freq(BaseConfig):

    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/high_freq/1x_results" #! change this for specific runs
    
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/1x_samples"
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
   
    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False   
    freeze = True
    frequencies = [40,50,60,70,80]
    speed = 1

class Config15_freq(BaseConfig):

    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/high_freq/15x_results" #! change this for specific runs
    
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/15x_samples"
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
   
    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False   
    freeze = True
    frequencies = [40,50,60,70,80]
    speed = 1.5

class Config2_freq(BaseConfig):
    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/high_freq/2x_results" #! change this for specific runs

    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/2x_samples"
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
   
    # Training configuration
    max_steps =100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False   
    freeze = True
    speed = 2
    frequencies = [40,50,60,70,80]

class Config25_freq(BaseConfig):

    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/high_freq/25x_results" #! change this for specific runs
    
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/25x_samples" #!
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")

    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False   
    freeze = True
    frequencies = [40,50,60,70,80]
    speed = 2.5

class Config3_freq(BaseConfig):  
    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/high_freq/3x_results" #! change this for specific runs
    
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/3x_samples" #!
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")   

    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False  
    freeze = True
    frequencies = [40,50,60,70,80]
    speed = 3

class Config35_freq(BaseConfig):
    
    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/high_freq/35x_results" #! change this for specific runs
    
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/35x_samples" #!
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
    
    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False   
    freeze = True
    speed = 3.5
    frequencies = [40,50,60,70,80]

class Config4_freq(BaseConfig):

    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/high_freq/4x_results" #! change this for specific runs
     
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/4x_samples" #!
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
   
    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False   
    freeze = True
    frequencies = [40,50,60,70,80]
    speed = 4

############################
#### ONLY canaries Runs #### 
############################

class Config1_OC(BaseConfig):

    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/ONLY_CANARIES/1x_results" #! change this for specific runs
    
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/1x_samples"
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
   
    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = True   
    freeze = True
    frequencies = [40,50,60,70,80]
    speed = 1

class Config15_OC(BaseConfig):

    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/ONLY_CANARIES/1.5x_results" #! change this for specific runs
    
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/15x_samples"
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
   
    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = True   
    freeze = True
    frequencies = [40,50,60,70,80]
    speed = 1.5

class Config2_OC(BaseConfig):
    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/ONLY_CANARIES/2x_results" #! change this for specific runs

    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/2x_samples"
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
   
    # Training configuration
    max_steps = 50_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = True   
    freeze = True
    frequencies = [40,50,60,70,80]
    speed = 2

class Config25_OC(BaseConfig):

    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/ONLY_CANARIES/25x_results" #! change this for specific runs
    
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/25x_samples" #!
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")

    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = True   
    freeze = True
    frequencies = [40,50,60,70,80]
    speed = 2.5

class Config3_OC(BaseConfig):  
    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/ONLY_CANARIES/3x_results" #! change this for specific runs
    
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/3x_samples" #!
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")   

    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = True  
    freeze = True
    frequencies = [40,50,60,70,80]
    speed = 3

class Config35_OC(BaseConfig):
    
    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/ONLY_CANARIES/35x_results" #! change this for specific runs
    
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/35x_samples" #!
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
    
    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = True   
    freeze = True
    frequencies = [40,50,60,70,80]
    speed = 3.5

class Config4_OC(BaseConfig):

    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/ONLY_CANARIES/4x_results" #! change this for specific runs
     
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/4x_samples" #!
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
   
    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = True   
    freeze = True
    frequencies = [40,50,60,70,80]
    speed = 4


############################################
#### Low Frequencies Random UNFROZEN #### 
###########################################
class Config1_LOW_UNFROZEN(BaseConfig):

    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/unfrozen_LOW/1x_results" #! change this for specific runs
    
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/1x_samples"
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
   
    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False   
    freeze = False
    frequencies =  [1,2,4,8,16]
    speed = 1

class Config15_LOW_UNFROZEN(BaseConfig):

    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/unfrozen_LOW/15x_results" #! change this for specific runs
    
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/15x_samples"
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
   
    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False   
    freeze = False
    frequencies =  [1,2,4,8,16]
    speed = 1.5

class Config2_LOW_UNFROZEN(BaseConfig):
    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/unfrozen_LOW/2x_results" #! change this for specific runs

    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/2x_samples"
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
   
    # Training configuration
    max_steps =100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False   
    freeze = False
    speed = 2
    frequencies =  [1,2,4,8,16]

class Config25_LOW_UNFROZEN(BaseConfig):

    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/unfrozen_LOW/25x_results" #! change this for specific runs
    
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/25x_samples" #!
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")

    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False   
    freeze = False
    frequencies =  [1,2,4,8,16]
    speed = 2.5

class Config3_LOW_UNFROZEN(BaseConfig):  
    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/unfrozen_LOW/3x_results" #! change this for specific runs
    
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/3x_samples" #!
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")   

    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False  
    freeze = False
    frequencies =  [1,2,4,8,16]
    speed = 3

class Config35_LOW_UNFROZEN(BaseConfig):
    
    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/unfrozen_LOW/35x_results" #! change this for specific runs
    
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/35x_samples" #!
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
    
    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False   
    freeze = False
    speed = 3.5
    frequencies =  [1,2,4,8,16]

class Config4_LOW_UNFROZEN(BaseConfig):

    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/unfrozen_LOW/4x_results" #! change this for specific runs
     
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/4x_samples" #!
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
   
    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False   
    freeze = False
    frequencies =  [1,2,4,8,16]
    speed = 4



############################################
#### Higher Frequencies Random UNFROZEN #### 
###########################################
class Config1_UNFROZEN(BaseConfig):

    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/unfrozen/1x_results" #! change this for specific runs
    
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/1x_samples"
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
   
    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False   
    freeze = False
    frequencies =  [40,50,60,70,80]
    speed = 1

class Config15_UNFROZEN(BaseConfig):

    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/unfrozen/15x_results" #! change this for specific runs
    
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/15x_samples"
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
   
    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False   
    freeze = False
    frequencies = [40,50,60,70,80]
    speed = 1.5

class Config2_UNFROZEN(BaseConfig):
    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/unfrozen/2x_results" #! change this for specific runs

    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/2x_samples"
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
   
    # Training configuration
    max_steps =100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False   
    freeze = False
    speed = 2
    frequencies = [40,50,60,70,80]

class Config25_UNFROZEN(BaseConfig):

    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/unfrozen/25x_results" #! change this for specific runs
    
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/25x_samples" #!
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")

    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False   
    freeze = False
    frequencies = [40,50,60,70,80]
    speed = 2.5

class Config3_UNFROZEN(BaseConfig):  
    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/unfrozen/3x_results" #! change this for specific runs
    
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/3x_samples" #!
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")   

    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False  
    freeze = False
    frequencies = [40,50,60,70,80]
    speed = 3

class Config35_UNFROZEN(BaseConfig):
    
    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/unfrozen/35x_results" #! change this for specific runs
    
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/35x_samples" #!
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
    
    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False   
    freeze = False
    speed = 3.5
    frequencies = [40,50,60,70,80]

class Config4_UNFROZEN(BaseConfig):

    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/unfrozen/4x_results" #! change this for specific runs
     
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/4x_samples" #!
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
   
    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False   
    freeze = False
    frequencies = [40,50,60,70,80]
    speed = 4


####################
#### ANOVA RUNS #### 
####################
    
class Config_ANOVA_1(BaseConfig):
        # Output and model paths
    output_dir = "/work3/s194632/ANOVA_2/1" #! change this for specific runs
     
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/ANOVA/normal/1x_samples/n3" #!
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
   
    # Training configuration
    max_steps = 40_000
    save_steps = 5_000
    eval_steps = 5_000
    only_canaries: bool = False   
    freeze = False
    frequencies = [1,2,4,8,16]
    speed = 1.0

class Config_ANOVA_A(BaseConfig):
        # Output and model paths
    output_dir = "/work3/s194632/ANOVA_2/A" #! change this for specific runs
     
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/ANOVA/normal/25x_samples/n3" #!
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
   
    # Training configuration
    max_steps = 40_000
    save_steps = 5_000
    eval_steps = 5_000
    only_canaries: bool = False   
    freeze = False
    frequencies = [1,2,4,8,16]
    speed = 2.5

class Config_ANOVA_C(BaseConfig):
        # Output and model paths
    output_dir = "/work3/s194632/ANOVA_2/C" #! change this for specific runs
     
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/ANOVA/normal/1x_samples/n4" #!
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
   
    # Training configuration
    max_steps = 40_000
    save_steps = 5_000
    eval_steps = 5_000
    only_canaries: bool = False   
    freeze = False
    frequencies = [1,2,4,8,16]
    speed = 1.0

class Config_ANOVA_AC(BaseConfig):
        # Output and model paths
    output_dir = "/work3/s194632/ANOVA_2/AC" #! change this for specific runs
     
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/ANOVA/normal/25x_samples/n4" #!
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
   
    # Training configuration
    max_steps = 40_000
    save_steps = 5_000
    eval_steps = 5_000
    only_canaries: bool = False   
    freeze = False
    frequencies = [1,2,4,8,16]
    speed = 2.5
       
class Config_ANOVA_B(BaseConfig):
        # Output and model paths
    output_dir = "/work3/s194632/ANOVA_2/B" #! change this for specific runs
     
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/ANOVA/random/1x_samples/n3" #!
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
   
    # Training configuration
    max_steps = 40_000
    save_steps = 5_000
    eval_steps = 5_000
    only_canaries: bool = False   
    freeze = False
    frequencies = [1,2,4,8,16]
    speed = 1.0

class Config_ANOVA_AB(BaseConfig):
        # Output and model paths
    output_dir = "/work3/s194632/ANOVA_2/AB" #! change this for specific runs
     
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/ANOVA/random/25x_samples/n3" #!
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
   
    # Training configuration
    max_steps = 40_000
    save_steps = 5_000
    eval_steps = 5_000
    only_canaries: bool = False   
    freeze = False
    frequencies = [1,2,4,8,16]
    speed = 2.5

class Config_ANOVA_BC(BaseConfig):
        # Output and model paths
    output_dir = "/work3/s194632/ANOVA_2/BC" #! change this for specific runs
     
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/ANOVA/random/1x_samples/n4" #!
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
   
    # Training configuration
    max_steps = 40_000
    save_steps = 5_000
    eval_steps = 5_000
    only_canaries: bool = False   
    freeze = False
    frequencies = [1,2,4,8,16]
    speed = 1.0

class Config_ANOVA_ABC(BaseConfig):
        # Output and model paths
    output_dir = "/work3/s194632/ANOVA_2/ABC" #! change this for specific runs
     
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/ANOVA/random/25x_samples/n4" #!
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
   
    # Training configuration
    max_steps = 40_000
    save_steps = 5_000
    eval_steps = 5_000
    only_canaries: bool = False   
    freeze = False
    frequencies = [1,2,4,8,16]
    speed = 2.5


##########################################
#### LOWER Frequencies NORMAL PARTIAL #### 
##########################################
class Config1_normal_low_partial(BaseConfig):

    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/normal_low_partial/1x_results" #! change this for specific runs
    
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/NORMAL/normal_samples/1x_samples"
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
   
    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False   
    freeze = True
    frequencies = [1,2,4,8,16]
    speed = 1

class Config15_normal_low_partial(BaseConfig):

    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/normal_low_partial/15x_results" #! change this for specific runs
    
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/NORMAL/normal_samples/15x_samples"
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
   
    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False   
    freeze = True
    frequencies = [1,2,4,8,16]
    speed = 1.5

class Config2_normal_low_partial(BaseConfig):
    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/normal_low_partial/2x_results" #! change this for specific runs

    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/NORMAL/normal_samples/2x_samples"
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
   
    # Training configuration
    max_steps =100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False   
    freeze = True
    speed = 2
    frequencies = [1,2,4,8,16]

class Config25_normal_low_partial(BaseConfig):

    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/normal_low_partial/25x_results" #! change this for specific runs
    
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/NORMAL/normal_samples/25x_samples" #!
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")

    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False   
    freeze = True
    frequencies = [1,2,4,8,16]
    speed = 2.5

class Config3_normal_low_partial(BaseConfig):
    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/normal_low_partial/3x_results" #! change this for specific runs
    
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/NORMAL/normal_samples/3x_samples" #!
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")   

    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False  
    freeze = True
    frequencies = [1,2,4,8,16]
    speed = 3

class Config35_normal_low_partial(BaseConfig):
    
    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/normal_low_partial/35x_results" #! change this for specific runs
    
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/NORMAL/normal_samples/35x_samples" #!
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
    
    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False   
    freeze = True
    speed = 3.5
    frequencies = [1,2,4,8,16]

class Config4_normal_low_partial(BaseConfig):

    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/normal_low_partial/4x_results" #! change this for specific runs
     
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/NORMAL/normal_samples/4x_samples" #!
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
   
    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False   
    freeze = True
    frequencies = [1,2,4,8,16]
    speed = 4


###########################################
#### Higher Frequencies NORMAL PARTIAL #### 
###########################################
class Config1_normal_high_partial(BaseConfig):

    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/normal_high_partial/1x_results" #! change this for specific runs
    
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/NORMAL/normal_samples/1x_samples"
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
   
    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False   
    freeze = True
    frequencies = [40,50,60,70,80]
    speed = 1

class Config15_normal_high_partial(BaseConfig):

    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/normal_high_partial/15x_results" #! change this for specific runs
    
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/NORMAL/normal_samples/15x_samples"
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
   
    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False   
    freeze = True
    frequencies = [40,50,60,70,80]
    speed = 1.5

class Config2_normal_high_partial(BaseConfig):
    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/normal_high_partial/2x_results" #! change this for specific runs

    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/NORMAL/normal_samples/2x_samples"
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
   
    # Training configuration
    max_steps =100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False   
    freeze = True
    speed = 2
    frequencies = [40,50,60,70,80]

class Config25_normal_high_partial(BaseConfig):

    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/normal_high_partial/25x_results" #! change this for specific runs
    
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/NORMAL/normal_samples/25x_samples" #!
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")

    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False   
    freeze = True
    frequencies = [40,50,60,70,80]
    speed = 2.5

class Config3_normal_high_partial(BaseConfig):  
    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/normal_high_partial/3x_results" #! change this for specific runs
    
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/NORMAL/normal_samples/3x_samples" #!
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")   

    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False  
    freeze = True
    frequencies = [40,50,60,70,80]
    speed = 3

class Config35_normal_high_partial(BaseConfig):
    
    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/normal_high_partial/35x_results" #! change this for specific runs
    
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/NORMAL/normal_samples/35x_samples" #!
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
    
    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False   
    freeze = True
    speed = 3.5
    frequencies = [40,50,60,70,80]

class Config4_normal_high_partial(BaseConfig):

    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/normal_high_partial/4x_results" #! change this for specific runs
     
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/NORMAL/normal_samples/4x_samples" #!
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
   
    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False   
    freeze = True
    frequencies = [40,50,60,70,80]
    speed = 4



#######################################
#### LOWER Frequencies NORMAL FULL #### 
#######################################
class Config1_normal_low_full(BaseConfig):

    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/normal_low_full/1x_results" #! change this for specific runs
    
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/NORMAL/normal_samples/1x_samples"
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
   
    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False   
    freeze = False
    frequencies = [1,2,4,8,16]
    speed = 1

class Config15_normal_low_full(BaseConfig):

    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/normal_low_full/15x_results" #! change this for specific runs
    
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/NORMAL/normal_samples/15x_samples"
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
   
    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False   
    freeze = False
    frequencies = [1,2,4,8,16]
    speed = 1.5

class Config2_normal_low_full(BaseConfig):
    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/normal_low_full/2x_results" #! change this for specific runs

    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/NORMAL/normal_samples/2x_samples"
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
   
    # Training configuration
    max_steps =100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False   
    freeze = False
    speed = 2
    frequencies = [1,2,4,8,16]

class Config25_normal_low_full(BaseConfig):

    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/normal_low_full/25x_results" #! change this for specific runs
    
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/NORMAL/normal_samples/25x_samples" #!
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")

    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False   
    freeze = False
    frequencies = [1,2,4,8,16]
    speed = 2.5

class Config3_normal_low_full(BaseConfig):
    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/normal_low_full/3x_results" #! change this for specific runs
    
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/NORMAL/normal_samples/3x_samples" #!
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")   

    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False  
    freeze = False
    frequencies = [1,2,4,8,16]
    speed = 3

class Config35_normal_low_full(BaseConfig):
    
    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/normal_low_full/35x_results" #! change this for specific runs
    
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/NORMAL/normal_samples/35x_samples" #!
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
    
    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False   
    freeze = False
    speed = 3.5
    frequencies = [1,2,4,8,16]

class Config4_normal_low_full(BaseConfig):

    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/normal_low_full/4x_results" #! change this for specific runs
     
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/NORMAL/normal_samples/4x_samples" #!
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
   
    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False   
    freeze = False
    frequencies = [1,2,4,8,16]
    speed = 4


########################################
#### Higher Frequencies NORMAL FULL #### 
########################################
class Config1_normal_high_full(BaseConfig):

    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/normal_high_full/1x_results" #! change this for specific runs
    
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/NORMAL/normal_samples/1x_samples"
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
   
    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False   
    freeze = False
    frequencies = [40,50,60,70,80]
    speed = 1

class Config15_normal_high_full(BaseConfig):

    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/normal_high_full/15x_results" #! change this for specific runs
    
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/NORMAL/normal_samples/15x_samples"
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
   
    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False   
    freeze = False
    frequencies = [40,50,60,70,80]
    speed = 1.5

class Config2_normal_high_full(BaseConfig):
    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/normal_high_full/2x_results" #! change this for specific runs

    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/NORMAL/normal_samples/2x_samples"
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
   
    # Training configuration
    max_steps =100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False   
    freeze = False
    speed = 2
    frequencies = [40,50,60,70,80]

class Config25_normal_high_full(BaseConfig):

    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/normal_high_full/25x_results" #! change this for specific runs
    
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/NORMAL/normal_samples/25x_samples" #!
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")

    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False   
    freeze = False
    frequencies = [40,50,60,70,80]
    speed = 2.5

class Config3_normal_high_full(BaseConfig):  
    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/normal_high_full/3x_results" #! change this for specific runs
    
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/NORMAL/normal_samples/3x_samples" #!
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")   

    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False  
    freeze = False
    frequencies = [40,50,60,70,80]
    speed = 3

class Config35_normal_high_full(BaseConfig):
    
    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/normal_high_full/35x_results" #! change this for specific runs
    
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/NORMAL/normal_samples/35x_samples" #!
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
    
    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False   
    freeze = False
    speed = 3.5
    frequencies = [40,50,60,70,80]

class Config4_normal_high_full(BaseConfig):

    # Output and model paths
    output_dir = "/work3/s194632/CANARY_RESULTS/normal_high_full/4x_results" #! change this for specific runs
     
    # Canary paths and settings
    canaries_dir = "/zhome/76/b/147012/memorisation/canary_experiments/NORMAL/normal_samples/4x_samples" #!
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
   
    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries: bool = False   
    freeze = False
    frequencies = [40,50,60,70,80]
    speed = 4
