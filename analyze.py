import matplotlib.pyplot as plt
from dataset import DatasetENIT
import numpy as np
import os
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from nltk.corpus import stopwords
import re

class Analyze:

    def __init__(self):
        self.dp = DatasetENIT()
        self.plot_path = "plots"


    def general_insights(self):

        file_path = "plots/general_insights.txt"
        if os.path.exists(file_path):
            os.remove(file_path)
   
        en_data, it_data = self.dp.get_raw_data()
        with open(file_path, 'a') as file:
            file.write(f"Number of Samples: {len(en_data)}\n")

        
        word_count_en = [ len(en_doc.split(" ")) for en_doc in en_data]
        word_count_it = [ len(it_doc.split(" ")) for it_doc in it_data]

        mat = np.array([word_count_en, word_count_it]).T
        mean = mat.mean(axis = 0)
        cov = (mat-mean).T @ (mat-mean) * (1/len(en_data))
        with open(file_path, 'a') as file:
            file.write("Covariance Matrix: (English - Italian)\n")
            file.write(str(cov))

        plt.scatter(word_count_en, word_count_it, color='blue', alpha=0.6, edgecolors='w', s=20, label = "Data Samples")
        plt.xlabel('English: Number of Words', fontsize=20)
        plt.ylabel('Italian: Number of Words', fontsize=20)
        max_value = max(max(word_count_en), max(word_count_it))
        plt.plot([0, max_value], [0, max_value], color='red', linestyle='--', label = "Correlation")
        plt.legend(fontsize = 10)
        plt.savefig(os.path.join(self.plot_path, "correlation_en_it.png"))

    def plot_bar(self, x, y, x_label, y_label, global_average, file):
        """
        Plots and saves a bar chart.

        Parameters:
        - x (list): The x-axis values.
        - y (list): The y-axis values.
        - x_label (str): The label for the x-axis.
        - y_label (str): The label for the y-axis.
        - title (str): The title of the plot.
        - global_average (float): The global average value to be indicated on the plot.
        - file (str): The filename for saving the plot.
        """
        plt.figure(figsize=(60, 60))
        plt.bar(x, y, color='slateblue')
        plt.xticks(fontsize=100)

        plt.yticks(
            np.arange(0, 1.5 * max(y), 10 ** np.ceil(np.log10(  max(1, ((1.5 * max(y)) // 10)  ))  )),
            fontsize=80,
        )

        plt.axhline(
            y=global_average,
            color="red",
            linestyle="--",
            linewidth=5,
            label="Average across all categoires",
        )

        max_height = max(y)
        plt.ylim(0, max_height * 2)

        plt.xlabel(x_label, fontsize=100)
        plt.ylabel(y_label, fontsize=100)
        plt.grid(linewidth=3)
        plt.legend(fontsize=100)
        plt.savefig(os.path.join(self.plot_path, file))


    def average_n_words_plot(self):
        """
        Plots and saves the average number of words per category.
        """
        en_data, it_data = self.dp.get_raw_data()

        words_sum = 0
        for en_doc in en_data:
            words_sum += len(en_doc.split(" "))

        avg_en = words_sum/len(en_data)

        words_sum = 0
        for it_doc in it_data:
            words_sum += len(it_doc.split(" "))

        avg_it = words_sum/len(it_data)

        self.plot_bar(["English", "Italian"], [avg_en, avg_it], "Languages", "Average number of words", (avg_en + avg_it)/2, "avg_num_words.png")


    def get_unique_words(self, text_list):
        # Create a translation table that maps each punctuation character to None
        translator = str.maketrans('', '', string.punctuation)
        
        # Use a set to collect unique words
        unique_words = set()
        
        for text in text_list:
            # Normalize the text: lower case and remove punctuation
            normalized_text = text.lower().translate(translator)
            # Split the text into words and update the set
            words = normalized_text.split()
            unique_words.update(words)
        
        return unique_words

    def n_unique_words_plot(self):
        """
        Plots and saves the number of unique words per category.
        """
        en_data, it_data = self.dp.get_raw_data()
  
        en_len = len(self.get_unique_words(en_data))
        it_len = len(self.get_unique_words(it_data))


        self.plot_bar(["English", "Italian"], [en_len, it_len], "Languages", "Number of unique words", (en_len + it_len)/2, "num_unique_words.png")


    def top_tfidf_words_plot(self):
        """
        Plots and saves the top TF-IDF words per category as word clouds.
        """

        en_data, it_data = self.dp.get_raw_data()
        english_stop_words = stopwords.words("english")
        italian_stop_words = stopwords.words("italian")
        for lang, data, stop_words in zip(["English", "Italian"], [en_data, it_data], [english_stop_words, italian_stop_words]):
            tfidf_vectorizer = TfidfVectorizer()
            mat = tfidf_vectorizer.fit_transform(data)
            avg_scores = mat.mean(axis=0).tolist()[0]
            terms = tfidf_vectorizer.get_feature_names_out()
            term_tfidf_scores = list(zip(terms, avg_scores))
            sorted_terms = sorted(term_tfidf_scores, key=lambda x: x[1], reverse=True)
            top_n = 500
            top_words = {
                term: score
                for term, score in sorted_terms[:top_n]
                if term not in stop_words
            }
            fig, ax = plt.subplots(figsize=(40, 40))
            wc = WordCloud(
                background_color="black",
                collocations=False,
                max_words=50,
                max_font_size=2000,
                min_font_size=8,
                width=800,
                height=1600,
                colormap=None,
            ).generate_from_frequencies(top_words)
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            plt.savefig(
                os.path.join(
                    self.plot_path,
                    f"top_tfidf_words_plot_{lang}.png",
                ),
                bbox_inches="tight",
            )

    def split_into_sentences(self, text):
        """
        Splits a given text into sentences using regex.

        Parameters:
        - text (str): The input text to split.

        Returns:
        - list: A list of sentences.
        """

        pattern = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)(?=\s|$)"
        sentences = re.split(pattern, text)
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

        return sentences


    def average_n_sentence_plot(self):
        """
        Plots and saves the average number of sentences per category.
        """
        
        en_data, it_data = self.dp.get_raw_data()
        avg_en = sum( [len(self.split_into_sentences(en_doc)) for en_doc in en_data])/len(en_data)
        avg_it = sum( [len(self.split_into_sentences(it_doc)) for it_doc in it_data])/len(it_data)

        self.plot_bar(["English", "Italian"], [avg_en, avg_it], "Languages", "Average number of Sentences", (avg_en + avg_it)/2, "avg_num_sents.png")


    def create_plots(self):
        self.general_insights()
        # self.average_n_words_plot()
        # self.n_unique_words_plot()
        # self.top_tfidf_words_plot()
        # self.average_n_sentence_plot()


def main():
    a = Analyze()
    a.create_plots()

if __name__ == "__main__":
    main()

