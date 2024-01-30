import pandas as pd

df = pd.read_csv('input/best_manual_scores_learning_cleaned.csv', index_col=0, header=0, sep=';')
# Columns: Diagnosis, Best scale, AUROC
print(df)

# Autism Spectrum Disorder
# ADHD-Combined Type
# Oppositional Defiant Disorder
# Any Diag
# Social Anxiety (Social Phobia)
# Generalized Anxiety Disorder
# Specific Learning Disorder with Impairment in Reading (test)
# Specific Learning Disorder with Impairment in Reading
# Specific Learning Disorder with Impairment in Written Expression (test)
# Specific Phobia
# NVLD without reading condition (test)
# NVLD (test)
# Specific Learning Disorder with Impairment in Mathematics (test)
# ADHD-Inattentive Type

diag_dict = {
    'Autism Spectrum Disorder': 'ASD',
    'ADHD-Combined Type': 'ADHD-C',
    'Oppositional Defiant Disorder': 'ODD',
    'Any Diag': 'Any',
    'Social Anxiety (Social Phobia)': 'SAD',
    'Generalized Anxiety Disorder': 'GAD',
    'Specific Learning Disorder with Impairment in Reading (test)': 'SLD-Read (test)',
    'Specific Learning Disorder with Impairment in Reading': 'SLD-Read',
    'Specific Learning Disorder with Impairment in Written Expression (test)': 'SLD-Write (test)',
    'Specific Phobia': 'SP',
    'NVLD without reading condition (test)': 'NVLD',
    'NVLD (test)': 'NVLD',
    'Specific Learning Disorder with Impairment in Mathematics (test)': 'SLD-Math (test)',
    'ADHD-Inattentive Type': 'ADHD-I'
}

df['Diagnosis'] = df['Diagnosis'].map(diag_dict)


# Plot a bar plot with AUROCs for diagnoses
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(df['Diagnosis'], df['AUROC'], color='blue')
ax.set_ylabel('AUROC')
ax.set_xlabel('Diagnosis')
ax.set_title('AUROC for sum scores')
ax.set_ylim([0.5, 1])
plt.xticks(rotation=45, ha="right", size=8)
plt.tight_layout()
plt.savefig('output/viz/sum_scores_aurocs.png')
plt.show()
plt.close()