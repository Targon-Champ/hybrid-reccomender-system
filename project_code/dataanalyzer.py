import pandas as pd
from .grapher import Grapher
class DataAnalyzer:
    """
    This class is specifically designed to analyze data for this project.
    It includes methods to perform various data analysis tasks.
    """
    def __init__(self):
        self.grapher = Grapher()
    
    def analyze_age_groups(self, users):
        age_labels = ['0-18', '19-25', '26-35', '36-45', '46-55', '56+']
        age_groups = pd.cut(users.get_data()['age'], bins=[0, 18, 25, 35, 45, 55, max(users.get_data()['age']+1)], labels=age_labels, right=True)
        age_group_counts = age_groups.value_counts().reindex(age_labels, fill_value=0)
        self.grapher.pie_chart(age_group_counts.index, age_group_counts.values, title="Age Groups Distribution")

    def analyze_gender_distribution(self, users):
        gender_counts = users.get_data()['gender'].value_counts()
        self.grapher.bar_graph(gender_counts.index, (gender_counts.values/sum(gender_counts.values))*100, title="Gender Distribution", xlabel="Gender", ylabel="Percentage")
        
    def analyze_occupation_distribution(self, users):
        occupation_counts = users.get_data()['occupation'].value_counts()
        self.grapher.pie_chart(occupation_counts.index, occupation_counts.values, title="Occupation Distribution")
    
    def analyze_ratings_distribution(self, ratings):
        rating_counts = ratings.get_data()['rating'].value_counts()
        self.grapher.bar_graph(rating_counts.index, (rating_counts.values/sum(rating_counts.values))*100, title="Ratings Distribution", xlabel="Ratings", ylabel="Percentage")
        
    def analyze_genre_distribution(self, items, genres):
        genre_counts = [sum(items.get_data()[col]) for col in genres]
        self.grapher.pie_chart(genres, genre_counts, title="Genre Distribution")
    