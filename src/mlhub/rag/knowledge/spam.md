# Spam detection

## What spam messages look like
Spam and phishing messages share recognisable signals: urgency ("act now", "final notice", "within 24 hours"), promises of prizes or free money, requests to call a number or text a short code, shortened or unusual links, requests to verify an account or password, and heavy use of capital letters and exclamation marks. Legitimate personal messages tend to be conversational, refer to shared context such as plans, people or places, and rarely contain links or premium-rate numbers.

## How the classifier works
The model turns each message into numbers with TF-IDF. Text is lowercased, split into words, common stop-words such as "the" and "and" are removed, and both single words and two-word phrases are counted. TF-IDF gives a high weight to words that are frequent in a message but rare across all messages, which is why words like "prize", "claim" or "urgent" stand out. A linear model, Naive Bayes or Logistic Regression, then learns which words point towards spam and which towards ham.

## Explaining a prediction
For a single message the explanation is the list of its words with the largest influence on the score. Words that push towards spam usually include claim, free, prize, winner, call, text and urgent. Words that push towards ham are everyday conversational words. The model only sees word patterns, so it does not understand intent and can be fooled by spam written in unusual wording.

## Precision, recall and the cost of mistakes
Precision is the fraction of messages flagged as spam that really are spam. Recall is the fraction of all spam that gets caught. A false positive means a legitimate message ends up in the spam folder, which is usually worse for users than letting an occasional spam message through, so real systems often tune the threshold for high precision. F1 balances precision and recall and is used here because only around 13 percent of messages in the SMS Spam Collection are spam, which makes plain accuracy misleading: always answering "ham" would already score about 87 percent.

## The SMS Spam Collection
The UCI SMS Spam Collection contains 5,574 English text messages labelled ham or spam, about 13 percent of them spam. Exact duplicate messages are removed before splitting into training and test sets so the same text cannot appear on both sides.

## Staying safe
Do not click links in unexpected messages, do not call back unknown numbers, and never share passwords or one-time codes. Banks and government agencies do not ask for credentials by text message. Report phishing to your mobile carrier or to the relevant national reporting service.
