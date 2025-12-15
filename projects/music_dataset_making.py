import pandas as pd

# Load dataset
df = pd.read_csv("C:/Users/devik/Downloads/music_features.csv")

# Function to assign mood based on audio features
def assign_mood(row):
    energy = row['energy']
    valence = row['valence']
    dance = row['danceability']
    loud = row['loudness']

    # Energetic songs — high energy + high loudness
    if energy > 0.65 and loud > -7:
        return "energetic"

    # Happy songs — moderate/high valence + danceability
    if valence > 0.55 and dance > 0.50:
        return "happy"

    # Calm songs — low energy + low loudness
    if energy < 0.40 and loud < -10:
        return "calm"

    # Sad songs — low valence + low danceability
    if valence < 0.40 and dance < 0.40:
        return "sad"

    # Default fallback rule
    # Pick closest mood using thresholds
    if valence >= 0.5:
        return "happy"
    if energy >= 0.5:
        return "energetic"
    if valence < 0.5 and energy < 0.5:
        return "sad"
    
    return "calm"


# Apply function
df['mood_auto'] = df.apply(assign_mood, axis=1)

# Save updated CSV
df.to_csv("music_features_with_mood.csv", index=False)

print("New CSV saved as: music_features_with_mood.csv")
print(df[['energy','valence','danceability','loudness','mood_auto']].head())




















