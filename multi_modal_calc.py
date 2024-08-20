import pickle
from sklearn.ensemble import RandomForestClassifier

random_forest_model =pickle.load(open("RFmodel.pkl", "rb"))

true_lable=[0,1,2,3,4]
def classify_soil(true_label=None):
    # Code to classify soil type and determine NPK ratio
    soil_type = true_label
    npk_ratio = {"Black Soil": (150, 37.5, 225), "Cinder Soil": (30, 12.5, 75), "Laterite Soil": (60, 25, 100),
                 "Peat Soil": (125, 50, 200), "Yellow Soil": (90, 37.5, 165)}
    return npk_ratio[soil_type]
def predict_crops(npk_ratio):

    input_data = [[*npk_ratio]]
    predicted_crops = random_forest_model.predict(input_data)
    return predicted_crops
def main():
    npk_ratio = classify_soil()
    predicted_crops_rf = predict_crops(npk_ratio)
    common_crops = predicted_crops_rf[:5]
    textual_model_predictions = top_results
    common_crops_textual = set(common_crops).intersection(textual_model_predictions)

    # Return the common 5 crops
    return list(common_crops_textual)

if __name__ == "__main__":
    top_results = main()
    print("Top results:", top_results)