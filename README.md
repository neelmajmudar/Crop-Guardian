# Welcome to Crop Guardian

In collboration with Lucas Fredonic

Built for farmers, gardeners, and everyone!
Powered by fine-tuned machine learning models,
- Upload an image of your crop
- Get a diagnosis and treatment plan in seconds

Try it out: https://cropguardian.streamlit.app

## Inspiration
Being surrounded by the agriculture and farmlands here at UC Davis, we were inspired to create a project that could help farmers and gardeners quickly diagnose their crops and receive treatment advice and more information from reputable sites. Recognizing the importance of sustainable farming practices in an ever-populating world, we believe that crop disease detection is a key part of shaping efficient food-growing practices and mitigating the spread of threatening diseases.

## What it does
We have created several machine learning models, specifically Convolutional Neural Networks (CNNs), that are capable of classifying diseases in fruits and vegetables, as well as determining the health of crops. A user can easily upload an image of their crop, and, in a matter of seconds, our models will provide a comprehensive analysis. The user receives a summary of the specific disease identified in their crop, along with a set of recommended preventive measures.

## How we built it
As the datasets differ greatly for each fruit/vegetable model, we had to create custom CNN models for each of them. We used Python and TensorFlow to code the bulk of the models, then trained them using IBM's Z. For the user interface, we used Streamlit.

## Challenges we ran into
We initially had trouble creating the different CNNs for the different models. As the datasets were not standardized, the model architectures had to vary greatly. We also come across several errors while processing and training our data.

## Accomplishments that we're proud of
- Successfully creating multiple CNN models for classifying diseases in fruits and vegetables, tailored to various datasets.
- Developing a user-friendly interface using Streamlit to make the application accessible to farmers and users.
- Providing users with detailed information about the specific disease detected in their crops and suggesting preventive measures.
## What we learned
- Extensive experience in building Convolutional Neural Network (CNN) models for image classification. Adapting model architectures to handle non-standardized datasets, demonstrating flexibility in machine learning applications.
- Gained insights into error handling and data processing during the training phase.
- Enhanced knowledge in deploying machine learning models with a user-friendly interface.
## What's next for Crop Rescue
We plan on adding more crop and disease options to our app. We also hope to improve the accuracy of disease classification by collecting larger and more diverse datasets. This will help the models become more robust in identifying diseases in more niche crop types. Finally, we would like to scale our application to a stronger website that can handle more users, and create a mobile interface for easier image uploading.
