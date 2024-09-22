from pydantic import BaseModel


# Modèle de données des utilisateurs
class User(BaseModel):
    username: str
    password: str


# Modèle de données pour les points
class Data_points(BaseModel):
    enterprise: dict
    Social: dict
    Environnement: dict
    Quality: dict
    Cost: dict
    Lead_time: dict
    Modernisation: dict
    ClientConsommateur: dict


# Définition de la structure de données attendue pour la requête POST
class Data_POST_request(BaseModel):
    question: str
    answer: str


# Modèle de données pour enregistrer les donnees d'acquisition dans mongodb
class Data_enterprise(BaseModel):
    company_name: str
    company_size: int
    activity_sector: str
