import pygame
from nine_mens_morris import NineMensMorris
from game_renderer import GameRenderer

def main():
    # Initialize game
    game = NineMensMorris()
    renderer = GameRenderer()
    
    clock = pygame.time.Clock()
    running = True
    
    print("Nine Men's Morris Game Started!")
    print("Rules:")
    print("1. Place 9 pieces each (Placing Phase)")
    print("2. Move pieces to adjacent positions (Moving Phase)")  
    print("3. When you have 3 pieces, you can fly to any empty position (Flying Phase)")
    print("4. Form mills (3 in a row) to remove opponent's pieces")
    print("5. Win by reducing opponent to 2 pieces or blocking all moves")
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    mill_formed = renderer.handle_click(game, event.pos)
                    
                    # Check for game over
                    game_over, winner = game.is_game_over()
                    if game_over:
                        print(f"Game Over! {'Red (Player 1)' if winner.value == 1 else 'Blue (Player 2)'} wins!")
        
        # Render game
        renderer.render(game)
        clock.tick(60)
    
    renderer.quit()

if __name__ == "__main__":
    main()