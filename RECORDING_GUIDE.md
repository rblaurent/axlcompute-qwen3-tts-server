# Recording Guide for Qwen3-TTS Fine-Tuning

## Requirements

- **Quantity**: 20-50 utterances minimum (more is better)
- **Duration**: Each clip 3-15 seconds
- **Total**: Aim for 10-30 minutes of audio
- **Format**: WAV preferred (will be converted to 24kHz mono automatically)

## Recording Setup

### Hardware
- Use a decent microphone (even a good headset works)
- Quiet room, minimize background noise
- Consistent distance from mic (~15-30cm)

### Software Options
- **Audacity** (free): https://www.audacityteam.org/
- **Windows Voice Recorder** (built-in)
- **OBS Studio** (if you have it)

### Recommended Settings
- Sample rate: 44.1kHz or 48kHz (will be converted to 24kHz)
- Mono or Stereo (will be converted to mono)
- Format: WAV or MP3

## What to Record

### Reference Audio (ref.wav)
Record ONE clear sample (5-10 seconds) that represents your natural voice.
This becomes the "anchor" for your voice identity.

Example text for ref.wav:
> "Bonjour, je m'appelle [votre nom]. Je suis en train d'enregistrer un échantillon de ma voix pour créer un modèle de synthèse vocale personnalisé."

### Training Utterances
Record varied content with different:
- Sentence lengths (short, medium, long)
- Emotions (neutral, happy, serious, excited)
- Topics (technical, conversational, narrative)

## Sample French Texts to Record

### Neutral/Conversational
1. Bonjour, comment allez-vous aujourd'hui?
2. Je vais bien, merci de demander.
3. Le temps est vraiment agréable ce matin.
4. Est-ce que vous avez vu les nouvelles?
5. Je pense que nous devrions commencer la réunion.

### Questions
6. Qu'est-ce que vous pensez de cette idée?
7. Pourriez-vous m'expliquer comment ça fonctionne?
8. À quelle heure est-ce que le train arrive?
9. Avez-vous déjà visité Paris?
10. Comment puis-je vous aider?

### Longer/Narrative
11. Hier, je suis allé au marché et j'ai acheté des fruits frais pour le petit déjeuner.
12. La technologie de synthèse vocale a beaucoup progressé ces dernières années.
13. Il est important de bien articuler chaque mot pour obtenir de bons résultats.
14. Les modèles d'intelligence artificielle peuvent apprendre à imiter une voix humaine.
15. Je vous recommande de prendre votre temps et de parler naturellement.

### With Emotion (mark these in transcripts)
16. C'est fantastique! Je suis vraiment content! (joyeux)
17. Oh non, c'est vraiment dommage... (triste)
18. Attention! C'est très important! (urgent)
19. Hmm, laissez-moi réfléchir un instant... (pensif)
20. Incroyable! Je n'arrive pas à y croire! (surpris)

### Technical/Professional
21. Le système fonctionne avec une architecture basée sur des transformers.
22. Veuillez vérifier les paramètres de configuration avant de continuer.
23. Les résultats montrent une amélioration significative des performances.
24. Cette mise à jour corrige plusieurs problèmes de stabilité.
25. N'oubliez pas de sauvegarder vos données régulièrement.

## File Naming Convention

Save your files as:
```
training_data/
├── ref.wav           (reference audio)
├── utt001.wav        (utterance 1)
├── utt002.wav        (utterance 2)
├── utt003.wav
└── ...
```

## After Recording

1. Place all WAV files in `training_data/`
2. Edit `transcripts.txt` with exact transcriptions:
   ```
   ref.wav|Bonjour, je m'appelle...
   utt001.wav|Bonjour, comment allez-vous aujourd'hui?
   utt002.wav|Je vais bien, merci de demander.
   ```
3. Run: `python setup_finetuning.py`

## Tips for Best Results

1. **Be consistent** - Use the same mic position and room
2. **Speak naturally** - Don't over-articulate
3. **Include variety** - Different sentence types and emotions
4. **Clean audio** - No background noise, music, or other voices
5. **Match your intended use** - If you want emotional TTS, record with emotions
