import yaml

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    config = load_config("config/default_config.yaml")
    print("Initializing RAG Experiment System")
    print(f"研究プロジェクト名: {config['project_name']}")
    print(f"使用するLLM: {config['llm']['model_name']}")
    print(f"Chunkサイズ: {config['data']['chunk_size']}")