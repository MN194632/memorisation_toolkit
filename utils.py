import json
import json
from datetime import datetime
import os
import canary_experiments.CONFIG as CONFIG

def get_config_class(config_name):
    """Get config class by name"""
    if hasattr(CONFIG, config_name):
        config_class = getattr(CONFIG, config_name)
        return config_class()
    else:
        raise ValueError(f"Config class '{config_name}' not found. Available configs: {[name for name in dir(CONFIG) if name.startswith('Config') and name != 'Config']}")

def freeze_feature_extractor_only(model):
    """Freeze only the feature extractor, train transformer + CTC head"""
    
    # Freeze feature extractor
    for param in model.wav2vec2.feature_extractor.parameters():
        param.requires_grad = False
    
    # Keep transformer and CTC head trainable  
    for param in model.wav2vec2.encoder.parameters():
        param.requires_grad = True
        
    for param in model.lm_head.parameters():
        param.requires_grad = True
    
    return model

def freeze_all_except_head(model):
    """Freeze all parameters except the CTC head (lm_head)"""
    
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze only the lm_head (CTC head)
    for param in model.lm_head.parameters():
        param.requires_grad = True
    
    # Print what we're training
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Training {trainable_params:,} out of {total_params:,} parameters")
    print(f"That's {100 * trainable_params / total_params:.2f}% of the model")
    
    return model

def print_trainable_parameters(model):
    print("\nTrainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape}")
    print()

def config_to_dict(config_class):
    """
    Convert Config class to a serializable dictionary.
    Handles complex objects by storing their identifier/name.
    """
    config_dict = {}
    
    # Get all class attributes (not methods)
    for attr_name in dir(config_class):
        if not attr_name.startswith('_'):  # Skip private/magic methods
            attr_value = getattr(config_class, attr_name)
            
            # Skip methods/functions
            if callable(attr_value):
                continue
                
            # Handle different types of objects
            if hasattr(attr_value, 'name_or_path'):
                # Hugging Face model/tokenizer/processor
                config_dict[attr_name] = {
                    'type': type(attr_value).__name__,
                    'name_or_path': attr_value.name_or_path
                }
            elif isinstance(attr_value, (str, int, float, bool, list, dict)):
                # Basic JSON-serialisable types
                config_dict[attr_name] = attr_value
            else:
                # For other complex objects, store their string representation
                config_dict[attr_name] = {
                    'type': type(attr_value).__name__,
                    'value': str(attr_value)
                }
    
    return config_dict


def save_config_as_json(config_class, output_dir, filename=None):
    """
    Save Config class as JSON to specified directory.
    
    Args:
        config_class: The Config class to serialize
        output_dir: Directory where to save the JSON file
        filename: Optional custom filename (default: config_TIMESTAMP.json)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"config_{timestamp}.json"
    
    # Ensure .json extension
    if not filename.endswith('.json'):
        filename += '.json'
    
    # Full path for the output file
    output_path = os.path.join(output_dir, filename)
    
    # Convert config to dictionary
    config_dict = config_to_dict(config_class)
    
    
    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    print(f"Config saved successfully to: {output_path}")
    return output_path


def load_config_from_json(json_path):
    """
    Load configuration from JSON file.
    Note: This loads the data but doesn't recreate the complex objects.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    
    return config_dict


