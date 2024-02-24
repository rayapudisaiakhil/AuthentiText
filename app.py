import streamlit as st
import matplotlib.pyplot as plt
from prediction import load_model, predict_label_and_extract_tfidf_words_colored
import seaborn as sns
# Define file paths for saved model and other necessary objects
model_file_path = 'random_forest_model.pkl'
tfidf_vectorizer_file_path = 'tfidf_vectorizer.pkl'
scaler_file_path = 'scaler.pkl'

# Load the trained model, TF-IDF vectorizer, and scaler
model, tfidf_vectorizer, scaler = load_model(model_file_path, tfidf_vectorizer_file_path, scaler_file_path)


def main():
    st.title('Authentitext')

    # Text input for the essay
    essay_text = st.text_area('Enter your essay here:', height=200)

    if st.button('Analyse Text'):
        if essay_text:
            # Make predictions
            label, important_words, highlighted_text, feature_values = predict_label_and_extract_tfidf_words_colored(essay_text, model, tfidf_vectorizer, scaler)

            # Display predicted label
            st.subheader('Prediction:')
            st.write(label)

            # Display highlighted text
            st.subheader('Text:')
            st.markdown(highlighted_text, unsafe_allow_html=True)

            # Determine the color based on the predicted label
            chart_color = 'lightgreen' if label == "The essay is written by a human." else 'lightcoral'

            # Display features as a bar plot
            st.subheader('Important Features:')
            # Normalize or scale feature values if necessary
            # Example: feature_values = {k: v * 100 for k, v in feature_values.items()}
            features = list(feature_values.keys())
            values = list(feature_values.values())

            fig, ax = plt.subplots()

            # Adjusting bar width for thinner bars
            bar_width = 0.3  # Decrease this value to make bars even thinner

            # Dynamically calculating positions for the bars
            bar_positions = range(len(features))

            # Plotting the bars with adjusted positions and specified width
            bars = ax.bar(bar_positions, values, width=bar_width, color=chart_color)

            # Adding the data value on head of each bar
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom', fontsize=8)

            # Styling the plot
            ax.set_ylabel('Values', fontsize=12, labelpad=10)
            ax.set_title('Feature Values', fontsize=14, pad=10)
            ax.set_xticks(bar_positions)
            ax.set_xticklabels(features, rotation=45, ha="right", fontsize=10)

            # Adjusting the Y-axis scale to go up to 100
            ax.set_ylim(0, 100)

            # Removing the top and right spines for a cleaner look
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Setting the face color to white for both plot and figure for a crisp look
            ax.set_facecolor('white')
            fig.set_facecolor('white')

            # Removing the grid
            ax.grid(False)

            # Ensuring layout is adjusted to prevent clipping of tick-labels or titles
            plt.tight_layout()

            # Displaying the bar plot
            st.pyplot(fig)

if __name__ == '__main__':
    main()