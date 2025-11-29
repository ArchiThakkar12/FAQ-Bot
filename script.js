// Simple sentence similarity function as fallback
class SimpleSimilarity {
    static getSimilarity(text1, text2) {
        const words1 = this.normalizeText(text1);
        const words2 = this.normalizeText(text2);
        
        const intersection = words1.filter(word => words2.includes(word));
        const union = [...new Set([...words1, ...words2])];
        
        return union.length > 0 ? intersection.length / union.length : 0;
    }
    
    static normalizeText(text) {
        return text.toLowerCase()
            .replace(/[^\w\s]/g, '')
            .split(/\s+/)
            .filter(word => word.length > 2);
    }
}

class FAQBot {
    constructor() {
        this.faqs = [];
        this.questionRephraser = null;
        this.answerFinder = null;
        this.isInitialized = false;
        this.useFallback = false;
        
        this.initializeBot();
    }

    async initializeBot() {
        try {
            this.updateStatus('Loading FAQs...');
            await this.loadFAQs();
            
            this.updateStatus('Loading AI models...');
            await this.loadModels();
            
            this.isInitialized = true;
            this.updateStatus('Ready to help!');
            
            // Enable input
            document.getElementById('user-input').disabled = false;
            document.getElementById('send-btn').disabled = false;
            
        } catch (error) {
            console.error('AI model loading failed, using fallback mode:', error);
            this.useFallback = true;
            this.isInitialized = true;
            this.updateStatus('Ready (Fallback Mode)');
            
            // Enable input even in fallback mode
            document.getElementById('user-input').disabled = false;
            document.getElementById('send-btn').disabled = false;
        }
    }

    async loadFAQs() {
        // Using embedded FAQs to avoid file loading issues
        this.faqs = [
            {
                "id": 1,
                "question": "What is your return policy?",
                "answer": "We offer a 30-day return policy for all unused items in original packaging. Returns are free within the US.",
                "category": "shipping_returns",
                "keywords": ["return", "refund", "exchange", "policy", "send back"]
            },
            {
                "id": 2,
                "question": "How long does shipping take?",
                "answer": "Standard shipping takes 3-5 business days. Express shipping takes 1-2 business days. International shipping takes 7-14 business days.",
                "category": "shipping_returns",
                "keywords": ["shipping", "delivery", "time", "arrive", "when", "how long"]
            },
            {
                "id": 3,
                "question": "Do you offer international shipping?",
                "answer": "Yes, we ship to over 50 countries worldwide. Shipping costs and times vary by destination.",
                "category": "shipping_returns",
                "keywords": ["international", "global", "countries", "overseas", "worldwide"]
            },
            {
                "id": 4,
                "question": "How can I track my order?",
                "answer": "You can track your order using the tracking link sent to your email or through your account dashboard. You'll also receive SMS updates if you provided a phone number.",
                "category": "orders",
                "keywords": ["track", "tracking", "status", "where is my order", "location"]
            },
            {
                "id": 5,
                "question": "What payment methods do you accept?",
                "answer": "We accept credit cards (Visa, MasterCard, American Express), PayPal, Apple Pay, and Google Pay.",
                "category": "payment",
                "keywords": ["payment", "pay", "credit card", "PayPal", "methods", "how to pay"]
            },
            {
                "id": 6,
                "question": "Can I change my order after placing it?",
                "answer": "You can modify your order within 1 hour of placement. After that, please contact customer service immediately.",
                "category": "orders",
                "keywords": ["change", "modify", "edit", "update", "order", "after"]
            },
            {
                "id": 7,
                "question": "Do you have a warranty?",
                "answer": "Yes, all products come with a 1-year manufacturer warranty. Extended warranties are available for purchase.",
                "category": "product",
                "keywords": ["warranty", "guarantee", "broken", "defect", "repair"]
            }
        ];
    }

    async loadModels() {
        try {
            // Check if transformers are available
            if (typeof pipeline === 'undefined') {
                throw new Error('Transformers not loaded');
            }

            // Try to load a very small model first
            this.updateStatus('Loading question understanding model...');
            this.questionRephraser = await pipeline(
                'text2text-generation',
                'Xenova/LaMini-Flan-T5-77M',
                { 
                    progress_callback: (progress) => {
                        if (progress.status === 'ready') {
                            this.updateStatus('Question model loaded, loading similarity model...');
                        }
                    }
                }
            );

            this.updateStatus('Loading similarity model...');
            this.answerFinder = await pipeline(
                'feature-extraction',
                'Xenova/all-MiniLM-L6-v2',
                {
                    progress_callback: (progress) => {
                        if (progress.status === 'ready') {
                            this.updateStatus('Models loaded successfully!');
                        }
                    }
                }
            );

            return true;
        } catch (error) {
            console.warn('Model loading failed, using fallback methods:', error);
            this.useFallback = true;
            return false;
        }
    }

    async rephraseQuestion(question) {
        if (this.useFallback || !this.questionRephraser) {
            // Simple cleaning for fallback
            return question.trim().replace(/\?/g, '').toLowerCase();
        }

        try {
            const prompt = `Make this question clear and formal: ${question}`;
            const result = await this.questionRephraser(prompt, {
                max_new_tokens: 60,
                temperature: 0.1,
                do_sample: false
            });
            
            return result[0].generated_text.trim();
        } catch (error) {
            console.error('Rephrasing failed, using original:', error);
            return question;
        }
    }

    async getEmbedding(text) {
        if (this.useFallback || !this.answerFinder) {
            return null;
        }

        try {
            const result = await this.answerFinder(text, {
                pooling: 'mean',
                normalize: true
            });
            return Array.from(result.data);
        } catch (error) {
            console.error('Embedding failed:', error);
            return null;
        }
    }

    cosineSimilarity(vecA, vecB) {
        if (!vecA || !vecB || vecA.length !== vecB.length) return 0;
        
        try {
            const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
            const normA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
            const normB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
            
            if (normA === 0 || normB === 0) return 0;
            return dotProduct / (normA * normB);
        } catch (error) {
            console.error('Similarity calculation error:', error);
            return 0;
        }
    }

    async findBestAnswer(question) {
        let candidates = [];
        const questionLower = question.toLowerCase();

        // Method 1: Try semantic similarity with embeddings
        if (!this.useFallback) {
            try {
                const questionEmbedding = await this.getEmbedding(question);
                if (questionEmbedding) {
                    for (const faq of this.faqs) {
                        const faqEmbedding = await this.getEmbedding(faq.question);
                        if (faqEmbedding) {
                            const similarity = this.cosineSimilarity(questionEmbedding, faqEmbedding);
                            if (similarity > 0.3) {
                                candidates.push({
                                    faq,
                                    similarity,
                                    type: 'semantic'
                                });
                            }
                        }
                    }
                }
            } catch (error) {
                console.error('Semantic search failed:', error);
            }
        }

        // Method 2: Keyword matching
        for (const faq of this.faqs) {
            let keywordScore = 0;
            let matchedKeywords = [];
            
            for (const keyword of faq.keywords) {
                if (questionLower.includes(keyword.toLowerCase())) {
                    keywordScore += 1;
                    matchedKeywords.push(keyword);
                }
            }
            
            // Normalize score
            if (keywordScore > 0) {
                keywordScore = keywordScore / faq.keywords.length;
                candidates.push({
                    faq,
                    similarity: keywordScore,
                    type: 'keyword',
                    matchedKeywords
                });
            }
        }

        // Method 3: Simple text similarity (fallback)
        if (candidates.length === 0) {
            for (const faq of this.faqs) {
                const similarity = SimpleSimilarity.getSimilarity(question, faq.question);
                if (similarity > 0.2) {
                    candidates.push({
                        faq,
                        similarity,
                        type: 'text_similarity'
                    });
                }
            }
        }

        // Remove duplicates and sort by similarity
        const uniqueCandidates = this.removeDuplicateCandidates(candidates);
        uniqueCandidates.sort((a, b) => b.similarity - a.similarity);

        const bestCandidate = uniqueCandidates[0];
        const needsClarification = uniqueCandidates.length > 1 && 
                                 uniqueCandidates[0].similarity < 0.7 &&
                                 uniqueCandidates[1].similarity > 0.4;

        return {
            answer: bestCandidate?.faq?.answer || this.getFallbackAnswer(),
            confidence: bestCandidate?.similarity || 0,
            candidates: uniqueCandidates.slice(0, 3),
            needsClarification
        };
    }

    removeDuplicateCandidates(candidates) {
        const seen = new Set();
        return candidates.filter(candidate => {
            const id = candidate.faq.id;
            if (seen.has(id)) return false;
            seen.add(id);
            return true;
        });
    }

    getFallbackAnswer() {
        return "I'm sorry, I couldn't find a specific answer to your question. Please try rephrasing or contact our support team for more assistance.";
    }

    async processQuestion(userQuestion) {
        if (!this.isInitialized) {
            return {
                type: 'error',
                message: "Bot is still initializing. Please wait a moment..."
            };
        }

        this.updateStatus('Processing your question...');
        
        try {
            const rephrasedQuestion = await this.rephraseQuestion(userQuestion);
            this.updateStatus('Searching for answers...');
            const result = await this.findBestAnswer(rephrasedQuestion);

            if (result.needsClarification && result.candidates.length > 1) {
                return this.askForClarification(result.candidates);
            }

            return this.formatAnswer(result.answer, result.confidence);
        } catch (error) {
            console.error('Processing error:', error);
            return {
                type: 'error',
                message: "I encountered an error while processing your question. Please try again."
            };
        }
    }

    askForClarification(candidates) {
        const clarificationMessage = "I found a few possible answers. Which one were you looking for?";
        const options = candidates.map((candidate, index) => 
            `${index + 1}. ${candidate.faq.question}`
        ).join('\n');
        
        return {
            type: 'clarification',
            message: `${clarificationMessage}\n\n${options}\n\nPlease respond with the number (1-${candidates.length}) that best matches your question.`,
            candidates: candidates
        };
    }

    formatAnswer(answer, confidence) {
        let confidenceText = "Here's what I found:";
        
        if (confidence > 0.8) {
            confidenceText = "I'm confident this answers your question:";
        } else if (confidence > 0.5) {
            confidenceText = "I think this might help:";
        } else {
            confidenceText = "This might be relevant to your question:";
        }
        
        return {
            type: 'answer',
            message: `${confidenceText}\n\n${answer}`,
            confidence: confidence
        };
    }

    updateStatus(message) {
        const statusElement = document.getElementById('status');
        if (statusElement) {
            statusElement.textContent = message;
        }
    }
}

// Chat Interface
class ChatInterface {
    constructor() {
        this.bot = new FAQBot();
        this.clarificationContext = null;
        this.setupEventListeners();
    }

    setupEventListeners() {
        const userInput = document.getElementById('user-input');
        const sendBtn = document.getElementById('send-btn');

        userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !userInput.disabled) {
                this.handleSendMessage();
            }
        });

        sendBtn.addEventListener('click', () => {
            if (!sendBtn.disabled) {
                this.handleSendMessage();
            }
        });
    }

    async handleSendMessage() {
        const userInput = document.getElementById('user-input');
        const sendBtn = document.getElementById('send-btn');
        const message = userInput.value.trim();

        if (!message) return;

        // Handle clarification responses
        if (this.clarificationContext && this.isNumericResponse(message)) {
            return this.handleClarificationResponse(message);
        }

        // Disable input while processing
        userInput.disabled = true;
        sendBtn.disabled = true;

        this.addMessage(message, 'user');
        userInput.value = '';

        // Show thinking message
        const thinkingId = this.addThinkingMessage();

        try {
            const response = await this.bot.processQuestion(message);
            
            // Remove thinking message
            this.removeMessage(thinkingId);
            
            if (response.type === 'clarification') {
                this.clarificationContext = response.candidates;
                this.addMessage(response.message, 'bot', 'clarification');
            } else if (response.type === 'error') {
                this.addMessage(response.message, 'bot');
            } else {
                this.clarificationContext = null;
                this.addMessage(response.message, 'bot');
                if (response.confidence !== undefined) {
                    this.addConfidence(response.confidence);
                }
            }

        } catch (error) {
            console.error('Error processing message:', error);
            this.removeMessage(thinkingId);
            this.addMessage("I'm sorry, I encountered an unexpected error. Please try again.", 'bot');
        } finally {
            // Re-enable input
            userInput.disabled = false;
            sendBtn.disabled = false;
            userInput.focus();
        }
    }

    handleClarificationResponse(message) {
        const choice = parseInt(message);
        const userInput = document.getElementById('user-input');
        const sendBtn = document.getElementById('send-btn');

        if (choice >= 1 && choice <= this.clarificationContext.length) {
            const selectedFAQ = this.clarificationContext[choice - 1].faq;
            
            this.addMessage(message, 'user');
            this.addMessage(selectedFAQ.answer, 'bot');
            this.addConfidence(this.clarificationContext[choice - 1].similarity);
            
            this.clarificationContext = null;
        } else {
            this.addMessage(message, 'user');
            this.addMessage("Please choose a valid number from the list.", 'bot');
        }

        userInput.value = '';
        userInput.focus();
    }

    isNumericResponse(message) {
        return /^\d+$/.test(message) && this.clarificationContext;
    }

    addMessage(message, sender, className = '') {
        const chatMessages = document.getElementById('chat-messages');
        const messageDiv = document.createElement('div');
        
        messageDiv.className = `message ${sender}-message ${className}`;
        messageDiv.textContent = message;
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        return messageDiv.id; // Return element id for removal
    }

    addThinkingMessage() {
        const chatMessages = document.getElementById('chat-messages');
        const messageDiv = document.createElement('div');
        
        messageDiv.id = 'thinking-message';
        messageDiv.className = 'message bot-message thinking';
        messageDiv.textContent = 'Thinking...';
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        return 'thinking-message';
    }

    removeMessage(messageId) {
        const element = document.getElementById(messageId);
        if (element) {
            element.remove();
        }
    }

    addConfidence(confidence) {
        const chatMessages = document.getElementById('chat-messages');
        const confidenceDiv = document.createElement('div');
        
        confidenceDiv.className = 'confidence';
        confidenceDiv.textContent = `Confidence: ${(confidence * 100).toFixed(1)}%`;
        
        chatMessages.appendChild(confidenceDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
}

// Initialize when page loads
let chatInterface;
document.addEventListener('DOMContentLoaded', () => {
    chatInterface = new ChatInterface();
});