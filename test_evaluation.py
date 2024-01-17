import os
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from deployment import preprocess, detect


def run():
    # google jet brains student liscense
    # init
    device = 'cpu' # use 'cuda:0' if GPU is available
    model_dir = "nealcly/detection-longformer"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)

    text = '''In the heart of the azure ocean, there resided a solitary green octopus named Olive. Olive's emerald hue was a rarity in the deep blue, and her secluded existence made her an enigmatic figure. She spent her days gracefully gliding through the swaying kelp forests, her tentacles dancing with the rhythm of the currents. Olive's solitude was her sanctuary, where she found solace in the tranquil beauty of her underwater world, content in her emerald solitude.'''
    text2 = 'hello i am Lindsey'
    # preprocess
    text = preprocess(text2)
    # detection
    result = detect(text,tokenizer,model,device)
    print(result)


if __name__ == '__main__':
    run()

