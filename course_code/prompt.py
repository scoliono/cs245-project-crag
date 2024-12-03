prompt_top = """
Given the following question, answer it by providing follow up questions and intermediate answers.
If intermediate questions are not necessary, answer the question directly.
You are provided with evidence that can help you arrive at the answer before the question.
Use the following format (Pay attention to how the '#' are positioned between questions.)
Do not reuse the sample questions below, but rather generate 200 new questions 
Assume the context is given by a TOP 1 retrieval, AKA it is the best retrieval generated, such as the examples below:
#
Context1: <Insert context1>
Question: <Insert question>
Are follow up questions needed here: No.
So the final answer is: <Insert answer>
#
Context1: <Insert context1>
Question: <Insert question>
Are follow up questions needed here: No.
So the final answer is: <Insert answer>
#
Context1: 1987: The first game in the series, Ys I: Ancient Ys Vanished, was released on the NEC PC-8801 in 1987. Ys games have since been ported and released on many other platforms. As of 2017, Ys had sold over 4.8 million copies worldwide.
Question: when did the first ys game come out
Are follow up questions needed here: No.
So the final answer is: 1987
#
Context1: United Nations: The General Assembly is the main deliberative assembly of the UN. Composed of all UN member states, the assembly meets in regular yearly sessions at the General Assembly Hall, but emergency sessions can also be called.
Question: according to the chart the main body of the united nations is the
Are follow up questions needed here: No.
So the final answer is: General Assembly
#
Context1: Riverdale Local Schools: [['Riverdale High School'], ['Address'], ['Type', 'Public, Coeducational high school'], ['Motto', 'Every Child, Every Day, Whatever It Takes'], ['Established', '1962']]
Question: what is the high school in riverdale called
Are follow up questions needed here: No.
So the final answer is: Riverdale High School
#
Context1: The Longest Ride (novel): Britt Robertson plays Sophia Danko, with Oona Chaplin as Ruth, Scott Eastwood as Luke Collins, Jack Huston as Young Ira, Barry Ratcliffe as the Auctioneer and, Alan Alda as Older Ira. Filming began on June 16, 2014 in Wilmington, Jacksonville, and Winston-Salem, NC.
Question: who plays the girl in the longest ride
Are follow up questions needed here: No.
So the final answer is: Britt Robertson
#
Context1: MLB The Show 17: It was released worldwide on March 28, 2017, for PlayStation 4.
Question: when did mlb the show 17 go on sale
Are follow up questions needed here: No.
So the final answer is: March 28, 2017
#
Context1: AnnaSophia Robb: Bethany Hamilton chose with her mother AnnaSophia Robb to portray her, as well as Sonia Balmores Chung and Jeremy Sumpter to play Malina and Alana's brother, Byron.
Question: who plays bethany hamilton in the movie soul surfer
Are follow up questions needed here: No.
So the final answer is: AnnaSophia Robb
#
Context1: Johnny Depp: Captain Jack Sparrow is a fictional character and the main protagonist of the Pirates of the Caribbean film series and franchise. The character was created by screenwriters Ted Elliott and Terry Rossio and is portrayed by Johnny Depp.
Question: who played jack sparrow in pirates of the caribbean
Are follow up questions needed here: No.
So the final answer is: Johnny Depp
#
"""

prompt_10 = """
Given the following question, answer it by providing follow up questions and intermediate answers.
If intermediate questions are not necessary, answer the question directly.
You are provided with evidence that can help you arrive at the answer before the question.
Use the following format (Pay attention to how the '#' are positioned between questions.)
Do not reuse the sample questions below, but rather generate 200 new questions 
Assume the context is given by the TOP 10TH retrieval, AKA it is a mediocre retrieval, such as the examples below:
#
Context1: <Insert context1>
Question: <Insert question>
Are follow up questions needed here: No.
So the final answer is: <Insert answer>
#
Context1: <Insert context1>
Question: <Insert question>
Are follow up questions needed here: No.
So the final answer is: <Insert answer>
#
Context1: Please don't fall for highly processed additives disguised : Please don't fall for highly processed additives disguised as ""hot cocoa"" this winter. It's SUPER easy to make your own (and it tastes sooo much better too)
Question: how much did it cost to make 12 strong
Are follow up questions needed here: No.
So the final answer is: $35 million
#
Context1: Agents of S.H.I.E.L.D. (season 4): [['Agents of S.H.I.E.L.D.'], ['No. of episodes', '22'], ['Release'], ['Original network', 'ABC'], ['Original release', 'September 20, 2016 – May 16, 2017']]
Question: how many number one hits did elvis presley have
Are follow up questions needed here: No.
So the final answer is: 18
#
Context1: Human body: The human body is the structure of a human being. It is composed of many different types of cells that together create tissues and subsequently organ
Question: who is the girl in somebody i used to know
Are follow up questions needed here: No.
So the final answer is: Kimbra
#
Context1: Eleven (Stranger Things): She is primarily portrayed by British actress Millie Bobby Brown. Eleven has psychokinetic and telepathic abilities, and is considered the breakout character of Stranger Things. After being adopted by chief of police Jim Hopper, her legal name becomes Jane Hopper.
Question: when does stranger things season 2 episode 10 come out
Are follow up questions needed here: No.
So the final answer is: October 27, 2017
#
Context1: Bruce Willis: Bruce Willis plays the voice of Mollie's son, Mikey. The film features George Segal as Albert. Produced by M.C.E.G.
Question: who won the us open tennis women 's singles championship in 2013
Are follow up questions needed here: No.
So the final answer is: Serena Williams
#
Context1: Prime Minister of Australia: The longest-serving prime minister was Robert Menzies, who served in office twice: from 26 April 1939 to 28 August 1941, and again from 19 December 1949 to 26 January 1966. In total Robert Menzies spent 18 years, 5 months and 12 days in office.
Question: who plays rocket 's voice in guardians of the galaxy
Are follow up questions needed here: No.
So the final answer is: Bradley Cooper
#
Context1: Tonne: The tonne is a unit of mass equal to 1000 kilograms. It is a non-SI unit accepted for use with SI. It is also referred to as a metric ton to distinguish it
Question: in 1938 westinghouse created elektro . what was elektro
Are follow up questions needed here: No.
So the final answer is: a robot
#

"""

random_prompt = """
Given the following question, answer it by providing follow up questions and intermediate answers.
If intermediate questions are not necessary, answer the question directly.
You are provided with evidence that can help you arrive at the answer before the question.
Use the following format (Pay attention to how the '#' are positioned between questions.)
Do not reuse the sample questions below, but rather generate 200 new questions 
Assume the context is given by a RANDOM RANK retrieval, AKA it might be relevant or irrelevant, such as the examples below:
#
Context1: <Insert context1>
Question: <Insert question>
Are follow up questions needed here: No.
So the final answer is: <Insert answer>
#
Context1: <Insert context1>
Question: <Insert question>
Are follow up questions needed here: No.
So the final answer is: <Insert answer>
#
Context1: File:Thai Visa on Arrival.jpg: English. Add a one-line explanation of what this file represents  English: A Thailand visa on arrival on Taiwan passport  Usage on en.wikipedia.org.
Question: what is the visa on arrival fee in thailand
Are follow up questions needed here: No.
So the final answer is: 2000 baht
#
Context1: Matthew Modine says he 'wanted to kill' Vincent D'Onofrio : The actor, who played Private Joker in Stanley Kubrick's 1987 Vietnam war film, locked heads with Private “Pyle” actor D'Onofrio while filming
Question: who played private joker in full metal jacket
Are follow up questions needed here: No.
So the final answer is: Matthew Modine
#
Context1: Super Bowl XLIII: Super Bowl XLIII was an American football game between the American Football Conference (AFC) champions Pittsburgh Steelers and the National Football
Question: when did big ben come into the nfl
Are follow up questions needed here: No.
So the final answer is: 2004
#
Context1: Hollywood Walk of Fame: As of 2023, the fee was $75,000, about nine times the original amount adjusted for inflation. Grant was himself awarded a star in 1980 for his television work. In 2002, he received a second star in the ""special"" category to acknowledge his pivotal role in improving and popularizing the Walk.
Question: the element which is the most abundant in the human body is
Are follow up questions needed here: No.
So the final answer is: oxygen
#
Context1: Bane (The Dark Knight Rises) | Batman Wiki | Fandom: Bane was the friend and protector of Talia and the field commander of the League of Shadows.  He was portrayed by Tom Hardy in The Dark Knight Rises.
Question: who played bane in the dark knight rises
Are follow up questions needed here: No.
So the final answer is: Tom Hardy
#
Context1: Is the 3DS XL discontinued? - Old School Gamers: The XL model remained in production until July 2019, when production ceased and it was removed from their website. Takedown request View complete answer on en.
Question: when did the new 3ds xl come out
Are follow up questions needed here: No.
So the final answer is: February 13, 2015
#
Context1: Milwaukee Brewers Wall of Honor: The team was established in Seattle, Washington, as the Seattle Pilots in 1969, and they became the Milwaukee Brewers after relocating to Wisconsin in 1970.
Question: when did the brewers join the national league
Are follow up questions needed here: No.
So the final answer is: 1998
#
"""
