import PySimpleGUI as sg
# import pandas to parse csv files
import pandas as pd
# using AutoTokenizer and AutoModelForSequenceClassification for the CardiffNLP model
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

def load_model():
    # model name
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    # load model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    # sentiment analysis pipeline using the model and tokenizer
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return sentiment_pipeline


def main():
    sg.theme('NeonYellow1')
    # layout of the GUI window
    layout = [
        # section for CSV file upload and analysis, uploads file from user directory path.
        [sg.Text("Select CSV File:"),
         sg.Input(key="-FILE-"),               # display the selected file path
         sg.FileBrowse(file_types=(("CSV Files", "*.csv"),))], # accept csv
        # main button load csv will call the model on the file or user can exit
        [sg.Button("Load CSV"), sg.Button("Exit")],
        [sg.Text("CSV Analysis Results:")],
        [sg.Multiline(key="-CSV_RESULTS-", size=(80, 15))],

        [sg.HorizontalSeparator()],

        # section for manual review input and analysis
        [sg.Text("Enter your review:")],
        [sg.Multiline(key="-USER_REVIEW-", size=(80, 5))],
        # when clicked by the user the model will run sentiment analysis on user input
        [sg.Button("Analyze Review")],
        [sg.Text("Review Analysis:")],
        [sg.Multiline(key="-USER_RESULTS-", size=(80, 5))]
    ]


    window = sg.Window("Sentiment Analysis", layout)

    # this variable holds the sentiment analysis pipeline, set to None initially.
    sentiment_pipeline = None
    # define mapping from model's labels to more definitive ones
    label_mapping = {
        "LABEL_0": "negative",
        "LABEL_1": "neutral",
        "LABEL_2": "positive"
    }
    # this event loop handles user actions on the gui as long as the window is running.
    while True:
        # read user actions as events and read values from the window
        event, values = window.read()
        if event in (sg.WINDOW_CLOSED, "Exit"):
            break

        if event == "Load CSV":
            # retrieve the file path entered or selected by the user
            file_path = values["-FILE-"]
            if file_path:
                try:

                    # use pandas dataframe to parse the csv file
                    df = pd.read_csv(file_path)
                    df = df.drop_duplicates(subset="review")
                    # get reviews from reviews column, if it does not exist, use the first column
                    if "review" in df.columns:
                        reviews = df["review"].tolist()
                    else:
                        reviews = df.iloc[:, 0].tolist()

                    # get ratings from rating column, if it does not exist print N/A
                    # for each review
                    if "rating" in df.columns:
                        ratings = df["rating"].tolist()
                    else:
                        ratings = ["N/A"] * len(reviews)

                    # load the sentiment analysis model
                    if sentiment_pipeline is None:
                        window["-CSV_RESULTS-"].update("Loading model, please wait...\n")
                        sentiment_pipeline = load_model()

                    # run sentiment analysis on the list of reviews
                    results = sentiment_pipeline(reviews)


                    # build the output string by iterating over each review and appending its sentiment result, and rating
                    csv_output = ""
                    for review, sentiment, rating in zip(reviews, results, ratings):
                        # replace original label with mapped label
                        mapped_label = label_mapping.get(sentiment["label"], sentiment["label"])
                        # create new dictionary that retains the original score
                        mapped_sentiment = {"label": mapped_label, "score": sentiment["score"]}
                        csv_output += f"Review: {review}\n"
                        csv_output += f"Overall Sentiment: {mapped_sentiment}\n"
                        csv_output += f"Rating: {rating}\n"
                        csv_output += "-" * 80 + "\n"

                    # update the bottom text field with the output strings
                    window["-CSV_RESULTS-"].update(csv_output)
                except Exception as e:
                    sg.popup_error(f"Error processing file: {e}")
            else:
                # if no file is selected, before clicking button, prompt the user to select a CSV file first
                sg.popup("Please select a CSV file first.")


        if event == "Analyze Review":
            # retrieve and strip any extra whitespace from the user's review text input
            user_review = values["-USER_REVIEW-"].strip()
            if user_review:
                try:
                    # load the sentiment analysis model
                    if sentiment_pipeline is None:
                        window["-USER_RESULTS-"].update("Loading model, please wait...\n")
                        sentiment_pipeline = load_model()

                    # run sentiment analysis on the user-entered review
                    result = sentiment_pipeline(user_review)[0]
                    mapped_label = label_mapping.get(result["label"], result["label"])
                    mapped_result = {"label": mapped_label, "score": result["score"]}

                    # build the output string for the manual review
                    user_output = f"Review: {user_review}\n"
                    user_output += f"Overall Sentiment: {mapped_result}\n"
                    user_output += "-" * 80 + "\n"

                    # update the bottom text field with the analysis output
                    window["-USER_RESULTS-"].update(user_output)
                except Exception as e:
                    sg.popup_error(f"Error analyzing review: {e}")
            else:
                sg.popup("Please enter a review to analyze.")
    window.close()


if __name__ == "__main__":
    main()
