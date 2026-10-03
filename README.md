# SetRecognition
 
## Step 1: Extract cards from raw images
Using cards_extraction.py
## Step 2: Rename extracted images according to their class
Manually. Class consists of 4 characters: 

Color - {'r', 'g', 'p'}(Red, Greed, Purple), 

Shape - {'d', 'w', 'o'}(Diamond, Wave, Oval), 

Filling - {'f', 'h', 'e'}(Full, Half, Empty),

Number - {'1', '2', '3'}

## Step 3: Synthetic Augmentation
Apply random transformation among:

Affine transformation,

Change in blur, brightness,

Applying random background

## Step 4: Create Training and Test Sets for Card Detection

## Step 5: Create simple CNN that gets a bbox and embeds into 12-dim vector(rgpdwofhe123)

## Step 6: From cards found, find set 
