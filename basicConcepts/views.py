from django.shortcuts import render

import tensorflow as tf
from tensorflow.keras.layers import TextVectorization



model_path = "D:\\nlp_project\\model_deploy\\model_5_acc_85_loss_41\\"

loaded_model = tf.keras.models.load_model(model_path)



import nltk
from nltk.tokenize import sent_tokenize
import tensorflow as tf

def nltk_function(abstract):
    # Tokenize sentences using NLTK
    abstract_lines = sent_tokenize(abstract)
    return abstract_lines


def split_chars(text):
    return ' '.join(list(text))


def make_predictions(text):
    classes = ['BACKGROUND', 'CONCLUSIONS', 'METHODS', 'OBJECTIVE', 'RESULTS']
    abstract_lines = list()

    abstract_lines = nltk_function(text)

    # Get total number of lines
    total_lines_in_sample = len(abstract_lines)

    # Go through each line in abstract and create a list of dictionaries containing features for each line
    sample_lines = []
    for i, line in enumerate(abstract_lines):
        sample_dict = {}
        sample_dict["text"] = str(line)
        sample_dict["line_number"] = i
        sample_dict["total_lines"] = total_lines_in_sample - 1
        sample_lines.append(sample_dict)

    # Get all line_number values from the sample abstract
    test_abstract_line_numbers = [line["line_number"] for line in sample_lines]

    # One-hot encode to the same depth as training data so that the model accepts the right input shape
    test_abstract_line_numbers_one_hot = tf.one_hot(test_abstract_line_numbers, depth=15)

    # Get all total_lines values from the sample abstract
    test_abstract_total_lines = [line["total_lines"] for line in sample_lines]

    # One-hot encode to the same depth as training data so that the model accepts the right input shape
    test_abstract_total_lines_one_hot = tf.one_hot(test_abstract_total_lines, depth=20)

    # Split abstract lines into characters
    abstract_chars = [split_chars(sentence) for sentence in abstract_lines]

    # Make predictions on sample abstract features
    test_abstract_pred_probs = loaded_model.predict(x=(test_abstract_line_numbers_one_hot,
                                                       test_abstract_total_lines_one_hot,
                                                       tf.constant(abstract_lines),
                                                       tf.constant(abstract_chars)))

    test_abstract_preds = tf.argmax(test_abstract_pred_probs, axis=1)

    test_abstract_pred_classes = [classes[i] for i in test_abstract_preds]
    
    
    
    BACKGROUND = ""
    CONCLUSIONS = ""
    METHODS = ""
    OBJECTIVE = ""
    RESULTS = ""

    for i, line in enumerate(abstract_lines):
        #print(f"{test_abstract_pred_classes[i]}: {line}")

        current_class = test_abstract_pred_classes[i]

        if current_class == "BACKGROUND":
            BACKGROUND += line
        elif current_class == "CONCLUSIONS":
            CONCLUSIONS += line
        elif current_class == "METHODS":
            METHODS += line
        elif current_class == "OBJECTIVE":
            OBJECTIVE += line
        elif current_class == "RESULTS":
            RESULTS += line
        

    return BACKGROUND, CONCLUSIONS, METHODS,  OBJECTIVE, RESULTS


def main(request):
    return render(request, 'index.html')

def home(request):
    return render(request, 'home.html')



def predict(request):
    text = request.GET['user_input']
    BACKGROUND, CONCLUSIONS, METHODS,  OBJECTIVE, RESULTS = "" , "" ,"" ,"", ""
    
    BACKGROUND, CONCLUSIONS, METHODS,  OBJECTIVE, RESULTS = make_predictions(text)
    print({
        'BACKGROUND' : BACKGROUND,
        'CONCLUSIONS' : CONCLUSIONS,
        'METHODS' : METHODS,
        'OBJECTIVE' : OBJECTIVE,
        'RESULTS' : RESULTS
        })
    return render(request, 'home.html',{
        'BACKGROUND' : BACKGROUND,
        'CONCLUSIONS' : CONCLUSIONS,
        'METHODS' : METHODS,
        'OBJECTIVE' : OBJECTIVE,
        'RESULTS' : RESULTS,
        'text':text
        })
    
    
   