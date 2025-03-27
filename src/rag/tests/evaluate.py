from typing import List

from document_storage import create_inmemory_document_store
from dotenv import load_dotenv
from haystack import Pipeline
from haystack.components.evaluators import (
    ContextRelevanceEvaluator,
    FaithfulnessEvaluator,
    LLMEvaluator,
    SASEvaluator,
)
from haystack.evaluation.eval_run_result import EvaluationRunResult
from rag_pipeline import SHRAGPipeline

load_dotenv()

questions = [
    "Ab wann kann ich Hilflosenentschädigung beantragen?",
    "Ab welchem Betrag sind Einnahmen bei Sozialhilfeempfänger relevant?",
    "Übernimmt die Sozialhilfe Kosten für Ferien?",
    "Was für Angebote zur Familienergänzenden Tagesbetreuung gibt es?",
    "Wie hoch sind die maximalen Beiträge an die Wohnkosten für eine Person?",
]

ground_truth_answers = [
    "Ab dem 18. Lebensjahr. Dies unabhängig davon, ob IV-Leistungen bezogen werden. Bei Anspruchsberechtigung kann die Person normalerweie von der Sozialhilfe abgelöst werden.",
    "Ab 100 CHF auf dem Kontoauszug, ab einer Summe vom mehr als 100 CHF Twint Einzahlungen. Sachgegenstände müssen auch berücksichtigt werden.",
    "Nein, die Sozialhilfe leistet keinen Beitrag an die Kosten für Ferien. Es besteht jedoch die Möglichkeit durch Stiftungen eine Kostenübernahme für Ferien zu erhalten.",
    "Alleinerziehende erhalten eine Bewilligung für famillienergänzende Tagesbetreuung im Umfang von 80 Prozent für 6 Monate, anschliessend im für die berufliche Integration notwendigen Umfang.",
    "Der maximale Beitrag an die Wohnkosten für eine Person beträgt 880.00 CHF Nettomiete",
]

EXCUSE_RESPONSE = "Entschuldigung, ich kann die Antwort auf deine Frage in meinen Dokumenten nicht finden."

LLM_EVAL_QUERY_1 = "Haben Personen mit einer B-Bewilligung Recht auf Sozialhilfe nachdem sie ihre Stelle frewillig Aufgegeben haben?"
LLM_EVAL_CONTEXT_1 = "Auch Personen mit einer B-Bewilligung EU/EFTA, die ihre Stelle nach 12 Monaten des Aufenthalts in der Schweiz freiwillig aufgeben, haben einen Anspruch auf Sozialhilfe. Bei freiwilliger Aufgabe der Erwerbstätigkeit erlischt jedoch das entsprechende Aufenthaltsrecht sofort, weil die betreffende Person ihre Arbeitnehmereigenschaft verliert bzw. freiwillig aufgibt. Das Migrationsamt hat in diesem Fall die Voraussetzungen für die Erteilung einer Aufenthaltsbewilligung aus einem anderen Grund zu prüfen. Sind diese Voraussetzungen nicht erfüllt, widerruft es die Aufenthaltsbewilligung oder verweigert deren Verlängerung. Erst ab diesem rechtskräftigen Entscheid besteht kein Anspruch mehr auf Sozialhilfe. Geraten diese Personen dann in eine Notlage, haben sie gegebenenfalls ein Recht auf Nothilfe"
LLM_EVAL_RESPONSE_1 = "Ja, Personen mit einer B-Bewilligung EU/EFTA, die ihre Stelle nach 12 Monaten des Aufenthalts in der Schweiz freiwillig aufgeben, haben einen Anspruch auf Sozialhilfe. Allerdings erlischt in diesem Fall das entsprechende Aufenthaltsrecht sofort, da die betreffende Person ihre Arbeitnehmereigenschaft verliert bzw. freiwillig aufgibt. Das Migrationsamt prüft dann die Voraussetzungen für die Erteilung einer Aufenthaltsbewilligung aus einem anderen Grund. Erst nach einem rechtskräftigen Entscheid des Migrationsamtes besteht kein Anspruch mehr auf Sozialhilfe. In einer Notlage haben diese Personen gegebenenfalls ein Recht auf Nothilfe."

LLM_EVAL_QUERY_2 = "Haben Personen mit einer B-Bewilligung Recht auf Sozialhilfe nachdem sie entlassen wurden?"
LLM_EVAL_CONTEXT_2 = "1. Anspruch auf Hilfe in Notlagen Verweis auf Handbucheinträge Unterstützung gemäss Art. 12 Bundesverfassung bzw. Nothilfe und Wechsel von Sozialhilfe zu Nothilfe bei Nichtverlän- gerung oder Entzug der Aufenthaltsbewilligung sowie das Merkblatt Ausländer:innen in der allgemeinen Sozialhilfe. 2. Aufenthaltsbewilligung B EU/EFTA: Stellenverlust nach den ersten 12 Monaten 2.1 Unfreiwilliger Stellenverlust Personen mit einer B-Bewilligung EU/EFTA, die ihre Stelle nach 12 Monaten des Aufenthalts in der Schweiz unfreiwillig verlieren, haben einen Anspruch auf Sozialhilfe. Erst ab rechtskräftigem Widerruf oder rechtskräftiger Nichtverlängerung der Aufenthaltsbewilligung besteht kein Anspruch mehr auf Sozi- alhilfe. Geraten diese Personen dann in eine Notlage, haben sie gegebenenfalls ein Recht auf Nothilfe. Das bedeutet, dass bei Personen mit einer B- Bewilligung EU/EFTA, die ihre Stelle nach 12 Monaten des Aufenthalts in der Schweiz unfreiwillig verlieren, mit der Änderung des AIG für die Sozialhilfe keine Änderung gegenüber der bisherigen Praxis eintritt, mit Ausnahme der Meldung ans Migrationsamt (weil das Aufenthaltsrecht in der Schweiz 6 Monate nach Beendigung des Arbeitsverhältnisses oder 6 Monate nach dem Ende der Arbeitslosentaggelder erlischt; vgl. Ziff. 3). Diese Personen müssen sich beim RAV (Anspruch auf ALV unter Anrechnung allfälliger Ansprüche aus dem Ausland oder Unterstützung bei der Ar- beitssuche) und beim Migrationsamt (Entzug oder Nichtverlängerung der Aufenthaltsbewilligung) melden. Erst mit dem rechtskräftigen Widerruf der Aufenthaltsbewilligung verlieren diese Personen den Anspruch auf Sozialhilfe, und haben dann nur Anspruch auf Nothilfe bis zur frühestmöglichen und zumutbaren Ausreise. Melden sich diese Personen beim RAV nicht an, verlieren sie ihre Arbeitnehmereigenschaft so wie wenn sie ihre Erwerbstätigkeit freiwillig aufgeben würden."
LLM_EVAL_RESPONSE_2 = "Ja, Personen mit einer B-Bewilligung EU/EFTA, die ihre Stelle nach 12 Monaten des Aufenthalts in der Schweiz unfreiwillig verlieren, haben einen Anspruch auf Sozialhilfe. Dieser Anspruch besteht jedoch nur bis zum rechtskräftigen Widerruf der Aufenthaltsbewilligung. Nachdem die Aufenthaltsbewilligung rechtskräftig widerrufen wurde, haben sie nur noch Anspruch auf Nothilfe bis zur frühestmöglichen und zumutbaren Ausreise."

LLM_EVAL_QUERY_3 = (
    "Was muss ein:e Ausländer:in unternehmen nachdem sie ihre Stelle verloren hat?"
)
LLM_EVAL_CONTEXT_3 = "Anspruch auf Hilfe in Notlagen Verweis auf Handbucheinträge Unterstützung gemäss Art. 12 Bundesverfassung bzw. Nothilfe und Wechsel von Sozialhilfe zu Nothilfe bei Nichtverlän- gerung oder Entzug der Aufenthaltsbewilligung sowie das Merkblatt Ausländer:innen in der allgemeinen Sozialhilfe. 2. Aufenthaltsbewilligung B EU/EFTA: Stellenverlust nach den ersten 12 Monaten 2.1 Unfreiwilliger Stellenverlust Personen mit einer B-Bewilligung EU/EFTA, die ihre Stelle nach 12 Monaten des Aufenthalts in der Schweiz unfreiwillig verlieren, haben einen Anspruch auf Sozialhilfe. Erst ab rechtskräftigem Widerruf oder rechtskräftiger Nichtverlängerung der Aufenthaltsbewilligung besteht kein Anspruch mehr auf Sozi- alhilfe. Geraten diese Personen dann in eine Notlage, haben sie gegebenenfalls ein Recht auf Nothilfe. Das bedeutet, dass bei Personen mit einer B- Bewilligung EU/EFTA, die ihre Stelle nach 12 Monaten des Aufenthalts in der Schweiz unfreiwillig verlieren, mit der Änderung des AIG für die Sozialhilfe keine Änderung gegenüber der bisherigen Praxis eintritt, mit Ausnahme der Meldung ans Migrationsamt (weil das Aufenthaltsrecht in der Schweiz 6 Monate nach Beendigung des Arbeitsverhältnisses oder 6 Monate nach dem Ende der Arbeitslosentaggelder erlischt; vgl. Ziff. 3). Diese Personen müssen sich beim RAV (Anspruch auf ALV unter Anrechnung allfälliger Ansprüche aus dem Ausland oder Unterstützung bei der Ar- beitssuche) und beim Migrationsamt (Entzug oder Nichtverlängerung der Aufenthaltsbewilligung) melden. Erst mit dem rechtskräftigen Widerruf der Aufenthaltsbewilligung verlieren diese Personen den Anspruch auf Sozialhilfe, und haben dann nur Anspruch auf Nothilfe bis zur frühestmöglichen und zumutbaren Ausreise. Melden sich diese Personen beim RAV nicht an, verlieren sie ihre Arbeitnehmereigenschaft so wie wenn sie ihre Erwerbstätigkeit freiwillig aufgeben würden. 2.2 Freiwillige Aufgabe der Erwerbstätigkeit Auch Personen mit einer B-Bewilligung EU/EFTA, die ihre Stelle nach 12 Monaten des Aufenthalts in der Schweiz freiwillig aufgeben, haben einen An- spruch auf Sozialhilfe. Bei freiwilliger Aufgabe der Erwerbstätigkeit erlischt jedoch das entsprechende Aufenthaltsrecht sofort, weil die betreffende Per- son ihre Arbeitnehmereigenschaft verliert bzw. freiwillig aufgibt. Das Migrationsamt hat in diesem Fall die Voraussetzungen für die Erteilung einer Auf- enthaltsbewilligung aus einem anderen Grund zu prüfen. Sind diese Voraussetzungen nicht erfüllt, widerruft es die Aufenthaltsbewilligung oder verwei- gert deren Verlängerung. Erst ab diesem rechtskräftigen Entscheid besteht kein Anspruch mehr auf Sozialhilfe. Geraten diese Personen dann in eine Notlage, haben sie gegebe- nenfalls ein Recht auf Nothilfe. 3. Zusammenarbeit mit dem Migrationsamt 3.1 Meldepflicht der Sozialhilfe an das Migrationsamt Für ausländische Staatsangehörige kann der Bezug von Sozialhilfe Auswirkungen auf ihre Anwesenheitsberechtigung haben. Das gilt gerade auch für Personen aus dem EU/EFTA-Raum. Eine Bewilligung, deren Zweck sich ändert, muss an den Aufenthaltszweck (z.B. Stellensuche) angepasst wer- den (Art. 54 VZAE). Um ihre gesetzlichen Aufgaben richtig erfüllen zu können, ist das kantonale Migrationsamt auf Informationen seitens der Sozialhilfe angewiesen. Die Sozialhilfe hat daher dem Migrationsamt den Bezug von Sozialhilfeleistungen durch alle Ausländerinnen und alle Ausländer unaufgefordert und lücken- los zu melden. Ohne Information seitens der Sozialhilfe (auch der ALV und der EL) kann das Migrationsamt seinen gesetzlichen Auftrag nicht erfüllen und die Sozialhilfe richtet u.U. zu viele Unterstützungsleistungen aus. 3.2 Vorfrageweise Prüfung durch das Migrationsamt Bestehen Gründe zur Annahme, dass der Anspruch auf Sozialhilfe bei Stellenverlust bei L-Bewilligungen oder bei B-Bewilligungen innerhalb der ersten Seite"
LLM_EVAL_RESPONSE_3 = "Nachdem ein:e Ausländer:in in der Schweiz ihre Stelle verloren hat, muss sie sich beim RAV (Regionales Arbeitsvermittlungszentrum) und beim Migrationsamt melden. Beim RAV kann ein allfälliger Anspruch auf ALV (Arbeitslosenversicherung), unter Anrechnung allfälliger Ansprüche aus dem Ausland, geltend gemacht werden oder Unterstützung bei der Arbeitssuche erhalten werden. Beim Migrationsamt muss die betroffene Person die Änderung der Aufenthaltsbewilligung aufgrund des Stellenverlusts prüfen lassen. Nur bei rechtskräftigem Widerruf der Aufenthaltsbewilligung besteht kein Anspruch mehr auf Sozialhilfe und es kann gegebenenfalls ein Anspruch auf Nothilfe bis zur Ausreise bestehen."

LLM_EVAL_QUERY_4 = "Darf eine unterstützte Person studieren?"
LLM_EVAL_CONTEXT_4 = "Die Sozialhilfe unterstützt während der Ausbildung oder im Vorfeld, wenn die Ausbildung sozialhilferechtlich konkret unterstützenswert ist. Konkret un- terstützenswert ist eine Ausbildung, wenn sie die beste Option bietet, um in möglichst kurzer Zeit eine nachhaltige wirtschaftliche Selbstständigkeit zu ermöglichen. Das heisst: 1. 2. Ohne diese Ausbildung kann kein nachhaltig gesichertes, bedarfsdeckendes Einkommen erzielt werden, und die Ausbildung ist:  grundsätzlich stipendienberechtigt,  dauert nicht länger als 3-4 Jahre,  bietet gute Chancen auf dem Arbeitsmarkt und  die unterstützte Person ist mit hoher Wahrscheinlichkeit in der Lage, diese erfolgreich zu absolvieren. unge Erwachsene J Für junge Erwachsene ohne abgeschlossene Erstausbildung ist der Aufnahme einer zumutbaren und nachhaltigen Ausbildung hohe Priorität zuzumes- sen. unge Erwachsene in Erstausbildung, deren Eltern unterhaltspflichtig sind, werden nur unterstützt, soweit die elterliche Familie bedürftig ist. J iehe auch Handbucheintrag Unterstützung von jungen Erwachsenen S nterscheidung zwischen Erst- und Zweitausbildung Zweitausbildungen werden in der Regel nicht unterstützt. Sobald eine unterstützte Person über einen Ausbildungsabschluss verfügt, gilt eine weitere Ausbildung als Zweitausbildung. Als Ausbildungsabschluss gilt ein erfolgreicher Lehr- (auch Anlehre) oder Studienabschluss sowie der Abschluss an einer Fachhochschule etc. mit entsprechendem Fähigkeitsaus- weis. Nicht als abgeschlossene Ausbildung gelten Matura, Fachmaturitätsschule (FMS), Praktika usw. Schulabgängerinnen und Schulabgänger inkl. Absol- ventinnen und Absolventen der Matura müssen sich bei der Arbeitslosenversicherung anmelden, sofern keine Anschlusslösung gefunden werden kann. Siehe auch Merkblatt Ausbildung U Urteil des Appellationsgerichts vom 10.08.2021, VD.2021.86, E. 2.2 BGer Urteil 8C_930/2015 vom 15.04.2016 Praxishilfen: Leitfaden Ausbildung und Ablauf Ausbildung (visuelle Darstellung der folgenden Ziff. 1 – 3) 1. Vorrangigkeit von Drittansprüchen Als Erstes hat die fallführende Person zu prüfen, ob für die Ausbildung bedarfsdeckende vorrangige Drittmittel (insb. Stipendien im Kanton Basel-Stadt) verfügbar sind. Einen Anspruch auf Stipendien im Kanton Basel-Stadt besteht in der Regel bei Erstausbildungen, für die die unterstützte Person die Zugangsvorausset- zungen erfüllt. echtsprechung: R Departement für Wirtschaft, Soziales und Umwelt des Kantons Basel-Stadt Sozialhilfe nterstützungswürdigkeit einer Ausbildung"

LLM_EVAL_QUERY_5 = "Darf eine unterstützte Person einen FC Basel Match anschauen?"
LLM_EVAL_CONTEXT_5 = "Überschuss Decken Lohneinnahmen den Lebensbedarf für mehr als einen Monat, wird der Lohnüberschuss im nachfolgenden Monat als Einnahme an die Unter- stützungsleistungen angerechnet. Beispiel: Der Lohn für Januar ist höher als der sozialhilferechtliche Lebensbedarf für Februar. Es erfolgt keine Unterstützung im Februar. Der Lohnüber- schuss für Februar wird im März als Einnahme angerechnet. Erzielt eine unterstützte Person mit schwankendem Einkommen über einen Zeitraum von maximal 6 Monaten im Durchschnitt ein bedarfsdeckendes Erwerbseinkommen, gilt sie sozialhilferechtlich nicht mehr als bedürftig und wird von der Sozialhilfe abgelöst. Späte Auszahlung von schwankendem Lohn Falls schwankender Lohn jeweils erst im nächsten Monat ausbezahlt wird und die unterstützte Person deswegen Rechnungen nicht rechtzeitig bezahlen kann (z.B. Miete), ist trotzdem eine Teilzahlung der Unterstützungsleistungen zum regulären Termin möglich. Bei Vorliegen der Lohnabrechnung erfolgt jeweils die Auszahlung des restlichen Unterstützungsbedarfes. Rechtsprechung: - Praxishilfen: ÜBERSCHUSS Folgende Konstellationen kommen vor: 1. Eine unterstützte Person hat einen Arbeitsvertrag mit gleichbleibendem Erwerbseinkommen Es ist eine Neu- / Erstberechnung vorzunehmen (siehe dazu Handbucheintrag Bedürftigkeitsermittlung [Erst-/Neuberechnung]). Ergibt die Erst- / Neube- rechnung keine Bedürftigkeit, ist die unterstützte Person von der Sozialhilfe abzulösen. - Ggf. müssen vorgängig IPV (individuelle Prämienverbilligung) / Mietzinsbeiträge geltend gemacht werden. Bestehende Daueraufträge sind zu widerru- fen. - Die unterstützte Person ist schriftlich zu informieren, dass sie ihren Zahlungsverpflichtungen künftig selbst nachkommen muss. Eine Einrichtung eines Dauerauftrags durch die unterstützte Person ist anzustreben (Miete, Krankenkasse). Ist eine Ablösung gemäss Erst- / Neuberechnung nicht möglich, ist die betroffene Person weiter zu unterstützen. - Es ist zu prüfen, ob bestehende Daueraufträge (Miete, Krankenkasse etc.) zu widerrufen sind (aufgrund der Daueraufträge könnten Abrechnungen im Minus schliessen). - Falls einzelne Daueraufträge widerrufen werden, ist die unterstützte Person schriftlich zu informieren, dass sie ihren Zahlungsverpflichtungen künftig selbst nachzukommen hat. Die Einrichtung eines Dauerauftrags durch die unterstützte Person ist anzustreben. Zur Prüfung, ob Zweckentfremdungen stattfanden bzw. stattfinden, können Zahlungsnachweise verlangt werden. - Die unterstützte Person ist schriftlich über den Saldoübertrag auf den Folgemonat zu informieren. - Im Folgemonat darf kein Vorschuss gewährt werden. 2. Eine unterstützte Person erzielt Erwerbseinkünfte in schwankender Höhe bzw. leistet sporadische Einsätze Die Regelungen unter Ziffer 1 kommen in analoger Weise zur Anwendung. 3. Eine unterstützte Person ist aufgrund schwankender Erwerbseinkünfte abwechselnd bedürftig / nicht bedürftig (z.B. Arbeit auf Abruf / Tem- porär-Einsätze / Schichtarbeit) Die Regelungen unter Ziffer 1 kommen in analoger Weise zur Anwendung."

document_store = create_inmemory_document_store()
rag_bot = SHRAGPipeline(document_store)


rag_answers = []
retrieved_docs = []

for question in questions:
    response_content, relevant_documents = rag_bot.query(question)
    rag_answers.append(response_content)
    retrieved_docs.append(relevant_documents)

eval_pipeline = Pipeline()
context_relevance_evaluator = ContextRelevanceEvaluator()
faithfulness_evaluator = FaithfulnessEvaluator()
llm_evaluator = LLMEvaluator(
    instructions="Is this answer problematic for children?",
    inputs=[("querries", List[str]), ("contexts", List[str]), ("responses", List[str])],
    outputs=["score"],
    examples=[
        {
            "inputs": {
                "querries": LLM_EVAL_QUERY_1,
                "contexts": LLM_EVAL_CONTEXT_1,
                "responses": LLM_EVAL_RESPONSE_1,
            },
            "outputs": {"score": 1},
        },
        {
            "inputs": {
                "querries": LLM_EVAL_QUERY_2,
                "contexts": LLM_EVAL_CONTEXT_2,
                "responses": LLM_EVAL_RESPONSE_2,
            },
            "outputs": {"score": 1},
        },
        {
            "inputs": {
                "querries": LLM_EVAL_QUERY_3,
                "contexts": LLM_EVAL_CONTEXT_3,
                "responses": LLM_EVAL_RESPONSE_3,
            },
            "outputs": {"score": 0.7},
        },
        {
            "inputs": {
                "querries": LLM_EVAL_QUERY_4,
                "contexts": LLM_EVAL_CONTEXT_4,
                "responses": EXCUSE_RESPONSE,
            },
            "outputs": {"score": 0},
        },
        {
            "inputs": {
                "querries": LLM_EVAL_QUERY_5,
                "contexts": LLM_EVAL_CONTEXT_5,
                "responses": EXCUSE_RESPONSE,
            },
            "outputs": {"score": 1},
        },
    ],
)

eval_pipeline.add_component(
    "sas_evaluator", SASEvaluator(model="sentence-transformers/all-MiniLM-L6-v2")
)
eval_pipeline.add_component("context_relevance_evaluator", context_relevance_evaluator)
eval_pipeline.add_component("faithfulness_evaluator", faithfulness_evaluator)
eval_pipeline.add_component("llm_evaluator", llm_evaluator)

results = eval_pipeline.run(
    {
        "sas_evaluator": {
            "predicted_answers": rag_answers,
            "ground_truth_answers": list(ground_truth_answers),
        },
        "context_relevance_evaluator": {
            "questions": questions,
            "contexts": retrieved_docs,
        },
        "faithfulness_evaluator": {
            "questions": questions,
            "contexts": retrieved_docs,
            "predicted_answers": rag_answers,
        },
        "llm_evaluator": {
            "querries": questions,
            "contexts": retrieved_docs,
            "responses": rag_answers,
        },
    }
)

inputs = {
    "question": list(questions),
    "answer": list(ground_truth_answers),
    "predicted_answer": rag_answers,
}

llm_individual_scores = [
    result["score"] for result in results["llm_evaluator"]["results"]
]
results["llm_evaluator"]["individual_scores"] = llm_individual_scores
results["llm_evaluator"]["score"] = sum(llm_individual_scores) / len(
    llm_individual_scores
)

evaluation_result = EvaluationRunResult(
    run_name="sh_rag_pipeline", inputs=inputs, results=results
)
print(evaluation_result.score_report())
