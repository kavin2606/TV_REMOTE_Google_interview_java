Say we have a television set-top box (DVR) and we want to search for a
show with an on-screen keyboard, e.g.:

  [A]B C D E   [A]B C D E F G
   F G H I J    H I J K L M N
   K L M N O    O P Q R S T U
   P Q R S T    V W X Y Z
   U V W X Y
   Z
   width=5      width=7

We want a function which takes an on-screen keyboard width and a show
title, and returns a string of remote-control button presses
(U/D/L/R/* (for Up, Down, Left, Right, and Select)):

   C++: string dvr_remote(int width, string title);
  Java: public String dvr_remote(int width, String title);
Python: def dvr_remote(width, title):

  For example, dvr_remote(5, "ER") could return:
    "RRRR*DDDLL*" or "RRRR*LLDDD*"

  - Cursor always starts on [A]
  - 1 <= width <= 26
  - title is all upper-case alpha
  - should return any valid in-bounds path (no wrapping)
  
