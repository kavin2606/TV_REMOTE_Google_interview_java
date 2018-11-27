import java.util.*;
public class remote {
    public static void main(String args[])
    {
        Scanner s = new Scanner(System.in);
        System.out.println("Please enter the name of the movie you want to navigate to");
        String movie =s.nextLine();
        System.out.println("Please enter the width of the remote");
        int width =s.nextInt();
        String directions = navigate(movie,width);
        System.out.println(directions);
    }
    public static String navigate(String movie,int width)
    {

        String directions="";
        int len=movie.length();
        movie = movie.toLowerCase();
        int current=97-97;
        int start=97-97;
        for(int i=0;i<len;i++)
        {
         int goal = movie.charAt(i)-97;
         int to=goal-start;
         int from=current-start;
         int todivisor=to/width;
         int toremainder=to%width;
         int fromdivisor=from/width;
         int fromremainder=from%width;
         int upcount,downcount,leftcount,rightcount;
         upcount=downcount=leftcount=rightcount=0;
         if(todivisor<fromdivisor)
         {
             upcount=fromdivisor-todivisor;
             downcount=0;
         }
         else if(todivisor>fromdivisor)
         {
             downcount=todivisor-fromdivisor;
             upcount=0;
         }
         else
         {
             upcount=0;
             downcount=0;
         }
         if(toremainder<fromremainder)
         {
             leftcount=fromremainder-toremainder;
             rightcount=0;
         }
         else if(toremainder>fromremainder)
         {
             rightcount=toremainder-fromremainder;
             leftcount=0;
         }
         else {
                leftcount=0;
                rightcount=0;
            }
         for(int j=0;j<upcount;j++)
         {
             directions=directions+"U";
         }
            for(int j=0;j<downcount;j++)
            {
                directions=directions+"D";
            }
            for(int j=0;j<leftcount;j++)
            {
                directions=directions+"L";
            }
            for(int j=0;j<rightcount;j++)
            {
                directions=directions+"R";
            }
            directions=directions+"*";
         current=goal;
        }

        return directions;
    }
}
