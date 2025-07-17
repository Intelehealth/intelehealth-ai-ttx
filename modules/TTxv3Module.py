import dspy

from signatures.TTxv3Signature import TTxv3Fields

class TTxv3Module(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(TTxv3Fields)

    def forward(self, case, diagnosis):
        print("Question in forward:")
        question = """What are the top 5 relevant Drug name-Strength-Route-Form, Dose, Frequency, Duration (number),Duration (units), Instruction (Remarks), Rationale for the medication for the patient given the diagnosis and patient case? Rank them in order of likelihood."""
        print(question)
        prediction = self.generate_answer(case=case, diagnosis=diagnosis, question=question)
        print(prediction)
        return dspy.Prediction(output=prediction) 