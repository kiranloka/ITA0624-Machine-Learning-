def classify_credit_score(score):
    if score < 300 or score > 850:
        return "Invalid credit score. Please enter a score between 300 and 850."
    elif score < 580:
        return "Poor"
    elif score < 670:
        return "Fair"
    elif score < 740:
        return "Good"
    elif score < 800:
        return "Very Good"
    else:
        return "Excellent"

def main():
    try:
        score = int(input("Enter your credit score (300-850): "))
        category = classify_credit_score(score)
        print(f"Credit Score: {score} → Category: {category}")
    except ValueError:
        print("Invalid input. Please enter a numeric value.")

if __name__ == "__main__":
    main()
