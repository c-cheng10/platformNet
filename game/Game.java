/**
 * Author: Zia Merchant
 * Date: May/June 2026
 * Subject: Main game implementation
 *
 * Sources:
 * - For collision detection: https://jeffreythompson.org/collision-detection/rect-rect.php
 * - For loop lessons (unit 3 - Iteration)
 * - IPO lessons (unit 1 - IPO)
 * - Moving plane lesson (unit 6 - GUI)
 * - Arrays (unit 4)
 */
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;

public class Game extends JPanel implements ActionListener, KeyListener {
    /** SETUP **/
    private Timer timer; // Timer for running actions
    private int level = 0; // The current level the player is at

    private double playerX = 10; // The player's x position
    private double playerY = 70; // The player's y position
    private double playerVelX = 0; // The player's x velocity
    private double playerVelY = 0; // The player's y velocity
    private double playerW = 5; // The player's width (pixels)
    private double playerH = 5; // The player's height (pixels)

    private boolean shouldMoveLeft = false; // Whether the player should be moving left
    private boolean shouldMoveRight = false; // Whether the player should be moving right
    private boolean shouldJump = false; // If the player should be trying to jump

    private boolean gameOver = false; // Tells whether the game is over

    private int deathCount = 0; // Amount of times the player has died in this game

    // Constant values
    private static double spikeHitboxWidth = 7; // How far out from the center of the spike to check for collisions

    // List of levels and the objects in them
    private static int[][][] levelData = {
        /*
         * Each ground object is stored as {x, y, width, height} because it is
         * a rectangle.
         * Each spike is stored as a point {x, y}, and collisions are detected
         * based on generated hitboxes.
         */

        // Level 1
        {
            {0,   400, 100, 100},   // Square one
            {150, 350, 100, 100}, // Second square
            {325, 300, 100, 100}, // Penultimate square
            {525, 275, 100, 100}, // Final square
        },
    };

    public Game() {
        timer = new Timer(10, this); // Instantiate the timer
        timer.start(); // Start the timer

        addKeyListener(this); // Add a key listener
        setFocusable(true); // Make this screen focusable

        setBackground(new Color(0xBCDFFF)); // Set the background to a light blue color
    }

    /** GRAPHICS **/
    public void paint(Graphics g) {
        super.paint(g); // Call this to reset the screen

        if (gameOver) {
            drawGameOver(g); // If the game is over, draw the gameover screen
        } else {
            drawGame(g); // If the game is not over, render the level and the player
        }
    }

    // Draw text centered horizontally on the screen
    private void drawCenteredText(Graphics g, String message, int y) {
        int width = g.getFontMetrics().stringWidth(message); // Calculate the width of the text
        int textX = (600/2 - width/2); // Calculate the position to center align the text

        g.drawString(message, textX, y); // Render the message on the screen
    }

    // Draw the screen showing the game over message
    private void drawGameOver(Graphics g) {
        g.setFont(new Font(Font.MONOSPACED, Font.BOLD, 32)); // Use a big font
        drawCenteredText(g, "Congrats! You beat the game!", 200); // Draw the congratulatory message

        g.setFont(new Font(Font.MONOSPACED, Font.BOLD, 16)); // Use a smaller font
        drawCenteredText(g, "Deaths: "+deathCount, 300); // Tell the player how many times they died
    }

    // Draw the level and player in the game
    private void drawGame(Graphics g) {
        int[][] currentLevel = levelData[level]; // Get the data for the current level

        // Get the spike image and resize it to 20x20 pixels
        ImageIcon spikeImg = new ImageIcon(
            new ImageIcon("spike.png").getImage().getScaledInstance(20, 20, Image.SCALE_DEFAULT)
        );

        ImageIcon playerImg = new ImageIcon("player.png"); // Get the player image

        for (int i = 0; i < currentLevel.length; i++) { // Iterate over each object in the level
            int[] object = currentLevel[i];

            if (object.length == 4) { // If the object is a rectangle, it is the ground
                // Set the color to black
                g.setColor(Color.BLACK);
                // Draw the ground on the screen
                g.fillRect(
                    object[0], // object top left x
                    object[1], // object top left y
                    object[2], // object width
                    object[3]  // object height
                );
            }

            // If the object is just a point, it is a spike
            if (object.length == 2) {
                spikeImg.paintIcon(this, g, object[0]-5, object[1]-5); // Draw the spike where the hitbox will be
            }
        }

        playerImg.paintIcon(this, g, (int)playerX, (int)playerY); // Draw the player image

        // Show the player which level they are on
        g.setFont(new Font(Font.MONOSPACED, Font.BOLD, 16)); // Set font to bold 16pt monospace
        g.drawString("Level "+(level+1), 20, 20);

        // If they are on the first level, show them the instructions for how to play
        if (level == 0) {
            g.drawString("Use the arrow keys to move!", 100, 100);
            g.drawString("Get to the right side of the screen -->", 100, 140);
        }
    }

    /** LOGIC **/

    // Checks if the player is touching the ground or a wall
    private boolean isTouchingGround() {
        // Source for collision detection: https://jeffreythompson.org/collision-detection/rect-rect.php
        for (int i = 0; i < levelData[level].length; i++) { // iterate over all parts of ground
            int[] object = levelData[level][i];
            if (object.length == 4) { // if the object isn't a rectangle, ignore it
                if (playerX + playerW >= object[0] &&   // right edge of player past left edge of object
                    playerX <= object[0] + object[2] && // left edge of player past right edge of object
                    playerY + playerH >= object[1] &&   // bottom edge of player below top edge of object
                    playerY <= object[1] + object[3]    // top edge of player above bottom edge of object
                ) {
                    return true;
                }
            }
        }

        return false;
    }

    // Checks if the player is touching a spike
    private boolean isTouchingSpike() {
        for (int i = 0; i < levelData[level].length; i++) {
            int[] object = levelData[level][i];
            if (object.length == 2) { // if the object is a spike, check for collisions
                if (playerX + playerW >= object[0] && // right edge of player past left side of spike
                    playerX <= object[0] + spikeHitboxWidth && // player left edge past spike right side
                    playerY + playerH >= object[1] && // player top edge past spike bottom
                    playerY <= object[1] + spikeHitboxWidth // player bottom edge past spike top
                ) {
                    return true;
                }
            }
        }

        return false;
    }

    // Checks if the player is dead
    private boolean isDead() {
        return (
            isTouchingSpike() ||
            playerY > 500 // Player has fallen off the map
        );
    }

    // Checks if we should go to the next level
    private boolean shouldChangeLevel() {
        // if the player is off-screen on the right side, we should go to the next level
        return playerX > 600-(playerW/2.0);
    }

    // Move the player based on physics and inputs
    private void updatePlayer() {
        if (shouldMoveLeft) {
            playerVelX -= 0.7; // If the left movement key is held, change their x velocity
        }
        if (shouldMoveRight) {
            playerVelX += 0.7; // If the right movement key is pressed, increase the x velocity
        }

        playerVelX *= 0.85; // Multiply for friction, so it doesn't move forever
        playerX += playerVelX; // Move the player by its velocity

        if (isTouchingGround()) { // If horizontal movement made a collision
            playerX -= playerVelX; // Then undo the movement
            playerVelX = 0; // And stop the player from colliding again
        }

        playerVelY += 0.3; // Increase y velocity for the gravity effect
        playerY += playerVelY; // Move the player up/down based on its velocity

        if (isTouchingGround()) { // Check if the vertical movement made a collision
            playerY -= playerVelY; // If it did, undo the movement
            
            // If we were falling, and touching the ground caused a collision, then stop falling
            if (playerVelY > 0) {
                playerVelY = 0;
                if (shouldJump) { // If we are touching the ground, and the player wants to jump
                    playerVelY = -8; // Set the y velocity to jump
                }
            } else { 
                playerVelY = 0; // If we were going up when we hit something, stop going up
            }
        }

        if (playerX < 0) { // If the player is going off-screen on the left side
            playerX = 0; // Move them back onto the screen
            playerVelX = 0; // Stop them from going off again
        }
    }

    // Reset all positions and velocities
    private void resetPlayer() {
        playerX = 10; // Set the x position to 10
        playerY = 350; // Set the y position to 350
        playerVelX = 0; // Set the x and y velocities to 0
        playerVelY = 0;
    }

    public void actionPerformed(ActionEvent e) {
        if (shouldChangeLevel()) { // If the player should advance to the next level
            level++; // Increase the level
            resetPlayer(); // Move the player to the starting position
        }

        if (level == levelData.length) { // If the player has finished the last level
            gameOver = true; // The game is then over
        }

        repaint(); // Draw whatever is needed based on the level the player is on

        if (!gameOver) { // If the game is not over
            updatePlayer(); // Update player position and physics

            // If the player dies, reset the level
            if (isDead()) {
                resetPlayer();
                deathCount++; // Also, increase the death count
            }
        }
    }

    /** INPUT **/

    public void keyPressed(KeyEvent e) {
        int code = e.getKeyCode(); // Get the key the player pressed

        if (code == KeyEvent.VK_LEFT) { // If they pressed the left arrow
            shouldMoveLeft = true; // Then they should move left
        }
        if (code == KeyEvent.VK_RIGHT) { // If they pressed the right arrow
            shouldMoveRight = true; // Then they should move right
        }
        if (code == KeyEvent.VK_UP) { // If they are pressing the up arrow
            shouldJump = true; // Then they are trying to jump
        }

        if (code == KeyEvent.VK_R) { // If the player presses the reset key
            resetPlayer(); // Move them back to the start
            gameOver = false; // Make sure the program knows to run the game
            level = 0; // Go back to the first level
            deathCount = 0; // Reset the death counter
        }
    }

    public void keyTyped(KeyEvent e) {
        // This function only exists to implement KeyListener
    }

    public void keyReleased(KeyEvent e) {
        int code = e.getKeyCode();

        if (code == KeyEvent.VK_LEFT) { // When they let go of the left arrow
            shouldMoveLeft = false; // They should stop moving left
        }
        if (code == KeyEvent.VK_RIGHT) { // When they let go of the right arrow
            shouldMoveRight = false; // They should stop moving right
        }
        if (code == KeyEvent.VK_UP) { // When they let go of the up arrow
            shouldJump = false; // They are no longer trying to jump
        }
    }
}
