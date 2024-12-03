import json
from flask import Flask, render_template, request
from pymongo import MongoClient
from transformers import pipeline
from pyngrok import ngrok

# Configuração do MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["Oraculo"]
rules_collection = db["Regras"]

# Modelos de NLP
qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
module_classifier = pipeline("text-classification", model="bert-base-uncased", return_all_scores=True)

# Flask App
app = Flask(__name__)

def get_context_by_module(module_name):
    """Busca o contexto do módulo no banco de dados."""
    module_data = rules_collection.find_one({"modulo": module_name})
    if module_data:
        return "\n".join(module_data["regras"])
    return None

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        question = request.form["question"]

        # Classificar a pergunta para identificar o módulo
        modules = list(rules_collection.find({}, {"modulo": 1}))
        module_names = [module["modulo"] for module in modules]
        classification = module_classifier(question)

        # Selecionar o módulo com maior score
        selected_module = max(classification, key=lambda x: x["score"])["label"]

        # Buscar o contexto do módulo
        context = get_context_by_module(selected_module)

        # Obter resposta usando a IA
        if context:
            result = qa_model(question=question, context=context)
            answer = result["answer"]
        else:
            answer = "Nenhuma regra encontrada para esse módulo."

        return render_template("index.html", question=question, answer=answer)

    return render_template("index.html", answer="")

@app.route("/update_context", methods=["POST"])
def update_context():
    """Atualiza o contexto de um módulo no banco de dados."""
    module_name = request.form["module"]
    new_rule = request.form["new_rule"]

    rules_collection.update_one(
        {"modulo": module_name},
        {"$push": {"regras": new_rule}},
        upsert=True
    )

    return "Contexto atualizado com sucesso!", 200

# Iniciando o ngrok e o servidor Flask
public_url = ngrok.connect(5000)
print(f"API está disponível em: {public_url}")

app.run(host="0.0.0.0", port=5000)
