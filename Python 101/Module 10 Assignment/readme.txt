1) Andrew Taylor atayl136
2) Module 10 Assignment: Debuggin K means due April 10 at 11:59PM EST
3) My approach was to step through the code mentally and I decided not to use pdb.
   I ran the code once just to see what happened, I took note that it stopped at an exception.
   I started at the top, reading the code line by line. I kept in mind the assignment specs from
   Module 5 so I knew what the objective was, and I also noted the comments for clues.
   I found many many logical errors to start, and then I reached the exception I noticed.
   I ran the code and it was still there. From there on out I ran the code with every bug fix,
   keeping track of exceptions and reading the code for any unflagged errors. I managed to find all the logical
   errors, the lexical and syntax errors, but actually found extra semantic errors because I chose to
   fix the code somewhat out of order, I feel. I deidn't notice the read mode was wrong until I had
   added an int() conversion, for instance. But, I think I caught them all.
   Where I am concerned is the output. It is close to but not exact to the model output. 
   Joe said this might happen. My code converges in 4 iterations while the model output is for 6.
   All in all I found at least the errors given, the code works, and provides a nice output.
   I did not change the algorithm.
4) No known bugs.