# Quiz App with Spaced Repetition

A React application for quiz practice with spaced repetition learning techniques. The app allows users to load quizzes from JSON files, take quizzes, get immediate feedback, and track their progress over time.

## Features

- **Quiz Loading**: Load quizzes from JSON files or use the sample quiz
- **One Question at a Time**: Focus on a single question
- **Immediate Feedback**: See if your answer is correct right away
- **Spaced Repetition**: Questions you struggle with appear more frequently
- **Multiple Choice Support**: Handle both single and multiple correct answers
- **Statistics Tracking**: Track your progress across different quizzes
- **Persistent Storage**: Your progress is saved locally

## Getting Started

1. Clone the repository
2. Install dependencies: `npm install`
3. Start the development server: `npm start`
4. Open [http://localhost:3000](http://localhost:3000) to view it in your browser

## Quiz JSON Format

The app expects quiz files in the following JSON format:

```json
{
  "title": "Quiz Title",
  "description": "Quiz description",
  "questions": [
    {
      "id": 1,
      "question": "Question text?",
      "answers": [
        {
          "answer": "Answer option 1",
          "correct": true
        },
        {
          "answer": "Answer option 2",
          "correct": false
        }
      ],
      "multiple": false
    }
  ]
}
```

- **title**: The title of the quiz
- **description**: A brief description of the quiz
- **questions**: An array of question objects:
  - **id**: Unique question identifier
  - **question**: The question text
  - **answers**: Array of answer options, each with:
    - **answer**: The answer text
    - **correct**: Boolean indicating if this is a correct answer
  - **multiple**: Boolean indicating if multiple answers can be selected

## Spaced Repetition Algorithm

The app uses a simplified version of the SuperMemo-2 spaced repetition algorithm:

- Questions you answer correctly will appear less frequently over time
- Questions you answer incorrectly will appear more frequently
- The system tracks your performance on each question and adjusts the review schedule accordingly