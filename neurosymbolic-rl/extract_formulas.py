"""
Extract the 200 formulas that were added to the feature bank.
"""
import re
import json

log_file = '/private/tmp/claude-501/-Users-tan-Desktop-Code-neurosymbolic-mnist-rl/tasks/b94ee56.output'

with open(log_file, 'r') as f:
    lines = f.readlines()

# Extract formulas that were added to bank
bank_formulas = []
for i, line in enumerate(lines):
    if '[Bank] Added to bank' in line:
        # Look backward to find the corresponding formula
        for j in range(i-1, max(0, i-10), -1):
            if 'Formula:' in lines[j]:
                formula_match = re.search(r'Formula:\s*(.+)', lines[j])
                # Find the accuracy
                for k in range(j-1, max(0, j-5), -1):
                    if 'Acc=' in lines[k]:
                        acc_match = re.search(r'Acc=([\d.]+)', lines[k])
                        if acc_match and formula_match:
                            acc = float(acc_match.group(1))
                            formula = formula_match.group(1).strip()
                            bank_formulas.append({
                                'formula': formula,
                                'accuracy': acc,
                                'tokens': formula.split()
                            })
                            break
                break

print(f"Extracted {len(bank_formulas)} formulas from bank")

# Save to file
with open('bank_formulas.json', 'w') as f:
    json.dump(bank_formulas, f, indent=2)

print(f"Saved to bank_formulas.json")

# Show statistics
if bank_formulas:
    accuracies = [f['accuracy'] for f in bank_formulas]
    print(f"\nBank Statistics:")
    print(f"  Count: {len(bank_formulas)}")
    print(f"  Mean Acc: {sum(accuracies)/len(accuracies):.3f}")
    print(f"  Max Acc: {max(accuracies):.3f}")
    print(f"  Min Acc: {min(accuracies):.3f}")
