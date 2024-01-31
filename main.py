import argparse
import multi, image_only, text_only
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model with specified hyperparameters')
    parser.add_argument('--model', type=str, help='Model type (multi, image_only, text_only)')
    args = parser.parse_args()

    model_type = args.model
    lr = args.lr
    dropout = args.dropout

    if model_type == 'multi':
        multi.main()
    elif model_type == 'image_only':
        image_only.main()
    elif model_type == 'text_only':
        text_only.main()
    else:
        print('Invalid model type. Please choose from multi, image_only, text_only.')