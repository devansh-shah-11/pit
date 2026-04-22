import matplotlib.pyplot as plt

data = [
    {'step': 100,  'overall': 0.28311425682507585, 'clean': 0.40080160320641284, 'adversarial': 0.2434077079107505},
    {'step': 200,  'overall': 0.3786653185035389,  'clean': 0.5871743486973948,  'adversarial': 0.30831643002028397},
    {'step': 300,  'overall': 0.3751263902932255,  'clean': 0.5591182364729459,  'adversarial': 0.31304935767410413},
    {'step': 400,  'overall': 0.36349848331648127, 'clean': 0.5531062124248497,  'adversarial': 0.299526707234618},
    {'step': 500,  'overall': 0.3574317492416582,  'clean': 0.5531062124248497,  'adversarial': 0.2914131169709263},
    {'step': 600,  'overall': 0.3564206268958544,  'clean': 0.533066132264529,   'adversarial': 0.29682217714672077},
    {'step': 700,  'overall': 0.34226491405460063, 'clean': 0.5130260521042084,  'adversarial': 0.2846517917511832},
]

steps      = [d['step'] for d in data]
overall    = [d['overall'] for d in data]
clean      = [d['clean'] for d in data]
adversarial = [d['adversarial'] for d in data]

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(steps, overall,     color='steelblue',  marker='o', label='Overall')
ax.plot(steps, clean,       color='seagreen',   marker='s', label='Clean')
ax.plot(steps, adversarial, color='tomato',     marker='^', label='Adversarial')

for i, (s, o, c, a) in enumerate(zip(steps, overall, clean, adversarial)):
    idx = i + 1
    ax.annotate(str(idx), (s, o), textcoords='offset points', xytext=(4, 4),  fontsize=7, color='steelblue')
    ax.annotate(str(idx), (s, c), textcoords='offset points', xytext=(4, 4),  fontsize=7, color='seagreen')
    ax.annotate(str(idx), (s, a), textcoords='offset points', xytext=(4, -10), fontsize=7, color='tomato')

ax.set_xlabel('Training Steps')
ax.set_ylabel('Exact-Match Accuracy')
ax.set_title('SFT Eval Accuracy vs Training Steps')
ax.set_xticks(steps)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('eval_accuracy.png', dpi=150)
plt.show()