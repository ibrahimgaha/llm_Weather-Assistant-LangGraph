# Projet 5 — Assistant météo avec LangGraph et NVIDIA AI

**Étudiant** Ibrahim Gaha

## Objectif

L’utilisateur pose une question sur la météo actuelle d’une ville. L’assistant
consulte Open-Meteo puis explique les conditions en langage simple. L’interface
affiche séparément la température, l’humidité, le vent et l’état du ciel.

## Architecture et rôle de LangGraph

Streamlit gère l’interface et conserve la conversation pendant la session.
`agent.py` définit un graphe LangGraph avec trois étapes qui partagent un état
contenant la question, la ville, les données météo et la réponse.

1. `extract_city` utilise un prompt LangChain et Mistral Large 4 pour trouver la ville.
   Une question de suivi peut reprendre la dernière ville résolue.
2. `get_weather` appelle `weather.py`. L’API de géocodage fournit les coordonnées.
   L’API Forecast d’Open-Meteo renvoie ensuite les conditions actuelles.
3. `generate_answer` transmet les données météo à NVIDIA pour produire une
   explication courte. Les mesures affichées proviennent directement d’Open-Meteo.

## Gestion des erreurs et résultats

Une ville inconnue, une réponse vide ou un délai dépassé déclenche un message
dans le chat. Les mesures manquantes sont indiquées comme indisponibles. Si
NVIDIA ne peut pas expliquer les conditions après leur récupération, les données
météo restent affichées.

Les tests hors ligne couvrent les appels Open-Meteo simulés, l’ordre des nœuds,
le contexte de suivi, les erreurs et le parcours Streamlit. La vérification
avec les services hébergés reste à effectuer depuis un environnement connecté.
