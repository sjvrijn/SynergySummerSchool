# SynergySummerSchool
Evolutionary Sequential Circuits at the Synergy Summer School 2018

Written for Python 3.6 (will fail on f-strings with any <=3.5)

Implements:
 - a namedtuple-based representation for logic gates that can be translated into a functional pyrtl circuit
 - some functionality to evaluate the correctness of that circuit compared to desired input/output sequences
 - a further translation of an integer-list representation into the aforementioned namedtuple-based
 representation
 - and a simple GA that can mutate the integer-list such that the mutated representation is still valid

The main 'trick' in terms of representing a sequential circuit is that the output or connection point of a
memory/register is encoded as a __negative__ number (index). When constructing the circuit from the representation,
the memories are created in advance before the rest of the circuit is made/connected column by column. This way,
any gate can refer to memory gates 'in the future' without disrupting the otherwise 'past-only' referencing model.
