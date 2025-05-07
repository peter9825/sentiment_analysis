import PySimpleGUI as sg
# import pandas to parse files
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
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


def main():
    sg.theme('NeonYellow1')  # set the GUI theme

    # initial default aspects to analyze within each review
    default_aspects = ["Battery life", "Design", "Performance"]

    # layout of GUI window
    layout = [
        [sg.Text("Select Data File (CSV, JSON, Excel):"),
         sg.Input(key="-FILE-"),
         sg.FileBrowse(file_types=(("Data Files", "*.csv *.json *.xlsx *.xls"),))],
        [sg.Button("Load Data"),
         sg.Text("Filter by sentiment:"),
         sg.Combo(["All", "positive", "neutral", "negative"], default_value="All", key="-FILTER-", readonly=True),
         sg.Button("Apply Filter"),
         sg.Button("Exit")],
        [sg.Text("Define aspects (comma-separated):"),
         sg.Input(default_text=", ".join(default_aspects), key="-ASPECTS-", size=(50,1)),
         sg.Button("Update Aspects")],
        [sg.Text("Analysis Results:")],
        [sg.Multiline(key="-RESULTS-", size=(80, 20))],
        [sg.HorizontalSeparator()],
        [sg.Text("Enter your review:")],
        [sg.Multiline(key="-USER_REVIEW-", size=(80, 5))],
        [sg.Button("Analyze Review")],
        [sg.Text("Review Analysis:")],
        [sg.Multiline(key="-USER_RESULTS-", size=(80, 10))]
    ]

    window = sg.Window("Product Review Analyzer", layout)

    sentiment_pipeline = None
    # mapping model output labels to human-readable strings
    label_mapping = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}
    # holds every review if unfiltered
    all_data = []
    # current list of aspects to analyze
    aspects = default_aspects.copy()

    # function to analyze sentiment for each product feature in a review
    def analyze_aspects(review_text):
        results = {}
        for aspect in aspects:
            # check if the aspect keywords are present and run sentiment analysis on them
            if aspect.strip().lower() in review_text.lower():
                text = f"{aspect}: {review_text}"
                res = sentiment_pipeline(text)[0]
                label = label_mapping.get(res['label'], res['label'])
                results[aspect] = {'label': label, 'score': res['score']}
            else:

                results[aspect] = {'label': 'N/A', 'score': None}
        return results

    # give overall analysis of dataset (mostly pos, neg, neu)
    def update_results(data):
        total = len(data)  #
        if total:
            counts = {"positive": 0, "neutral": 0, "negative": 0}
            for item in data:
                counts[item['label']] += 1
            pos_pct = counts['positive'] / total * 100
            neu_pct = counts['neutral'] / total * 100
            neg_pct = counts['negative'] / total * 100
            dominant = max(counts, key=lambda k: counts[k])
            summary = f"Overall: {pos_pct:.1f}% positive, {neu_pct:.1f}% neutral, {neg_pct:.1f}% negative — mostly {dominant} reviews"
        else:
            summary = "No reviews to summarize."

        # analyze user specified aspects of a product
        lines = [summary, '-' * 80]
        for item in data:
            lines.append(f"Review: {item['review']}")
            lines.append(f"Overall Sentiment: {{'label': '{item['label']}', 'score': {item['score']}}}")
            lines.append(f"Rating: {item['rating']}")
            lines.append("Aspect Sentiments:")
            # list each aspect's sentiment
            for aspect, ares in item['aspects'].items():
                score = ares['score'] if ares['score'] is not None else 'N/A'
                lines.append(f"  {aspect}: {{'label': '{ares['label']}', 'score': {score}}}")
            lines.append('-' * 80)
        window["-RESULTS-"].update("\n".join(lines))

    # event loop to handle user interactions
    while True:
        event, values = window.read()
        if event in (sg.WINDOW_CLOSED, "Exit"):
            break

        if event == "Update Aspects":
            user_input = values["-ASPECTS-"]
            aspects = [a.strip() for a in user_input.split(',') if a.strip()]
            sg.popup("Aspects updated:", ", ".join(aspects))

        elif event == "Load Data":
            file_path = values["-FILE-"]
            if not file_path:
                sg.popup("Please select a data file first.")
                continue
            try:
                lower = file_path.lower()
                # choose reader based on extension
                if lower.endswith('.csv'):
                    df = pd.read_csv(file_path)
                elif lower.endswith('.json'):
                    df = pd.read_json(file_path)
                elif lower.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(file_path)
                else:
                    raise ValueError("Unsupported file format: must be CSV, JSON, or Excel")

                # get review column or first column if 'review' not present
                if "review" in df.columns:
                    df = df.drop_duplicates(subset="review")
                    reviews = df["review"].astype(str).tolist()
                else:
                    df = df.drop_duplicates()
                    reviews = df.iloc[:, 0].astype(str).tolist()
                # get ratings or set to 'N/A'
                ratings = df.get("rating", pd.Series(["N/A"] * len(reviews))).tolist()

                if sentiment_pipeline is None:
                    window["-RESULTS-"].update("Loading model, please wait...\n")
                    sentiment_pipeline = load_model()
                all_data.clear()
                sentiments = sentiment_pipeline(reviews)
                for rev, sent, rate in zip(reviews, sentiments, ratings):
                    label = label_mapping.get(sent['label'], sent['label'])
                    aspects_res = analyze_aspects(rev)
                    all_data.append({'review': rev, 'label': label, 'score': sent['score'], 'rating': rate, 'aspects': aspects_res})
                # filter out reviews with no aspect mentions (if aspects defined)
                display = [item for item in all_data if any(v['label'] != 'N/A' for v in item['aspects'].values())] if aspects else all_data
                update_results(display)

            except Exception as e:
                sg.popup_error(f"Error processing file: {e}")

        # filter by sentiment (pos, neg, neu)
        elif event == "Apply Filter":
            if not all_data:
                sg.popup("No data loaded to filter.")
                continue
            filt = values["-FILTER-"]
            # filter dataset by overall sentiment
            filtered = all_data if filt == "All" else [i for i in all_data if i['label'] == filt]
            # then filter by aspect presence
            filtered = [item for item in filtered if any(v['label'] != 'N/A' for v in item['aspects'].values())] if aspects else filtered
            update_results(filtered)

        # analyze single review provided by user (keyboard input)
        elif event == "Analyze Review":
            text = values["-USER_REVIEW-"].strip()
            if not text:
                sg.popup("Please enter a review to analyze.")
                continue
            try:
                if sentiment_pipeline is None:
                    window["-USER_RESULTS-"].update("Loading model, please wait...\n")
                    sentiment_pipeline = load_model()
                res = sentiment_pipeline(text)[0]
                lbl = label_mapping.get(res['label'], res['label'])
                mapped = {'label': lbl, 'score': res['score']}
                aspects_res = analyze_aspects(text)
                out_lines = [f"Review: {text}", f"Overall Sentiment: {mapped}", "Aspect Sentiments:"]
                for aspect, ares in aspects_res.items():
                    score = ares['score'] if ares['score'] is not None else 'N/A'
                    out_lines.append(f"  {aspect}: {{'label': '{ares['label']}', 'score': {score}}}")
                out_lines.append('-' * 80)
                window["-USER_RESULTS-"].update("\n".join(out_lines))
            except Exception as e:
                sg.popup_error(f"Error analyzing review: {e}")
    window.close()


if __name__ == "__main__":
    main()
