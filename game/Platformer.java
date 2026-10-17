/**
 * Author: Zia Merchant
 * Date: May/June 2026
 * Subject: Startup for game class
 */
import javax.swing.JFrame;

public class Platformer extends JFrame {
    public Platformer() {
        Game g = new Game(); // Instantiate the game
        add(g); // Add the game to the JFrame
        setResizable(false); // Do not allow the user to resize the game
        setSize(100, 100); // Set the size to 100x100 pixels
        setTitle("Platformer"); // Set the window title to platformer
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); // Shut down the program when the window is closed
        setVisible(true); // Make the JFrame visible
        g.requestFocusInWindow(); // Give the input focus to the game
    }

    public static void main(String[] args) {
        new Platformer(); // Instantiate the platformer, starting a new game
    }
}
