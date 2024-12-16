from main import ClassificationAgent

def test_classification():
    # 定義標籤及其描述
    label2desc = {
        "apple": "A fruit that is typically red, green, or yellow.",
        "banana": "A long curved fruit that grows in clusters and has soft pulpy flesh and yellow skin when ripe.",
        "cherry": "A small, round stone fruit that is typically bright or dark red.",
    }

    # 測試輸入文本
    test_text = "The fruit is red and about the size of a tennis ball."

    # 初始化分類模型
    config = {
        "model": "Qwen/Qwen2.5-7B-Instruct",  # Specify the model name here
        "device": 1  # Use -1 for CPU, or specify GPU ID
    }
    agent = ClassificationAgent(config)

    # 執行分類
    predicted_label = agent(label2desc, test_text)

    # 打印結果
    print(f"Input Text: {test_text}")
    print(f"Predicted Label: {predicted_label}")

if __name__ == "__main__":
    test_classification()
