ML Debugging Practice for Interview
This repository contains realistic ML debugging problems to help you prepare for technical interviews. Each problem has multiple intentional bugs that mirror real-world issues you might encounter.

📁 Files
ml_debugging_practice.py - Contains 8 buggy ML implementations
test_ml_debugging.py - Comprehensive unit tests that pass only when bugs are fixed
README.md - This file with setup instructions
🚀 Quick Setup
1. Download Files
Save the three files to your local directory:

bash
# Create a new directory
mkdir ml_debugging_practice
cd ml_debugging_practice

# Copy the code files (from the artifacts above)
# ml_debugging_practice.py
# test_ml_debugging.py
# README.md
2. Install Dependencies
bash
pip install torch unittest2 psutil
3. Run Initial Tests (Should Fail)
bash
python test_ml_debugging.py
You should see many test failures - this is expected! Each failure points to a specific bug.

🎯 Practice Strategy
Phase 1: Bug Hunting (Day 1-2)
Read one problem at a time in ml_debugging_practice.py
Try to spot bugs by inspection before running tests
Run specific test classes to focus on one problem:
bash
python -m unittest test_ml_debugging.TestMultiHeadAttention -v
Phase 2: Systematic Debugging (Day 3-4)
Use test failures as hints - read the assertion messages carefully
Fix one bug at a time and re-run tests
Practice explaining your debugging process out loud
Phase 3: Interview Simulation (Day 5)
Have someone pick a random problem
Walk through your debugging methodology step-by-step
Explain what you're checking and why
🐛 Problems and Key Bugs
Problem 1: MultiHeadAttention
❌ Missing attention scaling factor
❌ Potential shape mismatches with different sequence lengths
Problem 2: PositionalEncoding
❌ Registered as parameter instead of buffer
❌ Gradient flow issues
Problem 3: Training Loop
❌ Wrong gradient operation order
❌ Memory leaks from storing tensor objects
❌ Incorrect loss accumulation
Problem 4: KV Cache
❌ Device mismatches
❌ Cache overflow handling
❌ Incorrect tensor indexing
Problem 5: Gradient Clipping
❌ Wrong function name
❌ Accumulating tensors instead of scalars
Problem 6: Transformer Decoder
❌ Wrong normalization order (post-norm vs pre-norm)
❌ Missing dropout in residual connections
Problem 7: Attention Masks
❌ Incorrect causal mask creation
❌ Broadcasting dimension errors
Problem 8: Model Evaluation
❌ Missing model.eval() call
❌ Missing torch.no_grad() context
❌ Non-deterministic results
🧪 Running Tests
Run All Tests
bash
python test_ml_debugging.py
Run Specific Test Class
bash
python -m unittest test_ml_debugging.TestMultiHeadAttention -v
Run Single Test Method
bash
python -m unittest test_ml_debugging.TestMultiHeadAttention.test_attention_scaling -v
💡 Debugging Tips for Interview
1. Systematic Approach
Start with shapes: "Let me trace through the tensor dimensions..."
Check common issues: "I should verify the gradient flow..."
Use test failures as guides: "This test suggests the issue is..."
2. Think Out Loud
Explain your reasoning: "I'm checking this because..."
State your hypotheses: "I suspect the problem is..."
Describe your next steps: "Next, I would..."
3. Common ML Bug Categories
Tensor Operations: Shape mismatches, wrong transposes
Memory Management: Gradient accumulation, device mismatches
Training Logic: Wrong operation order, missing mode switches
Numerical Stability: Missing normalization, overflow issues
🎯 Interview Day Checklist
Before the Interview
 Can spot 2-3 bugs per problem in 5-10 minutes
 Can explain debugging methodology clearly
 Familiar with common PyTorch gotchas
 Practiced thinking out loud
During the Interview
 Start with high-level approach: "First, I'd check..."
 Be systematic: trace through the code step-by-step
 Ask clarifying questions: "What symptoms are you seeing?"
 Explain your reasoning for each check
🔧 Common Fix Patterns
Memory Leaks
python
# ❌ Wrong - stores tensor with gradients
losses.append(loss)

# ✅ Correct - stores scalar value
losses.append(loss.item())
Gradient Operations
python
# ❌ Wrong order
loss.backward()
optimizer.step()
optimizer.zero_grad()

# ✅ Correct order
optimizer.zero_grad()
loss.backward()
optimizer.step()
Model Modes
python
# ❌ Missing evaluation setup
def evaluate(model, data):
    for batch in data:
        output = model(batch)

# ✅ Proper evaluation setup
def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        for batch in data:
            output = model(batch)
📚 Additional Resources
PyTorch Common Pitfalls
Transformer Implementation Guide
ML Debugging Best Practices
🎉 Success Criteria
You're ready for the interview when:

✅ All unit tests pass
✅ You can explain each bug and its fix
✅ You can spot bugs quickly by inspection
✅ You can articulate a systematic debugging approach
Good luck with your interview! 🚀

