# Agent with tools
The agent was extanded with a tool that can evaluate mathematical expressions. The tool is implemented using a function decorated with @tool, allowing agent to call it when a calculate is requiered. This enables the agent to perform precise computations instead of estimating results.

## Implementation:
A calculator tool was added to the agent using the calculate() function. The tool allows the agent to evaluate mathematical expressions. 

## Result:
The agent can now perform exact calculations by calling the tool instead of relying only on the language model.