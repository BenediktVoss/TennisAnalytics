from ultralytics import YOLO
import wandb
import os

##disable warnings
import warnings
warnings.filterwarnings("ignore")
os.environ["WANDB_API_KEY"] = "WANDB_API_KEY"

# training function
def train_yolo_model(config=None):
    with wandb.init(config=config) as run:
        config = wandb.config
        # read from config model_name, data_path, epochs, img_size, devices, batch_size, seed, box, cls, dfl, freeze
        model_name = config.model_name
        data_path = config.data_path
        epochs = config.epochs
        img_size = config.img_size
        devices = config.devices
        batch_size = config.batch_size
        seed = config.seed
        box = config.box
        cls = config.cls
        dfl = config.dfl

        try:
            print(f"Starting training for model: {model_name}")
            model = YOLO(model_name)  # Load the specified YOLO model

            results = model.train(
                data=data_path,
                epochs=epochs,
                imgsz=img_size,
                device=devices,
                box=box,
                cls=cls,
                dfl=dfl,
                batch=batch_size,
                project="player-ball-tracking-tuning",
                seed=seed,
                patience=4,
                name=f"train_{model_name.split('.')[0]}"  # Set a custom name for the training run
            )

            # validate the model
            metrics = model.val(data=data_path, imgsz=img_size, save_json=True, project="player-ball-tracking")

            # get metrics for both classes (person 1 and ball 0)
            metrics_total = metrics.results_dict
            results_person = metrics.class_result(1)
            results_ball = metrics.class_result(0)

            # calculate f1 score total and for both classes
            f1_total = 2 * (metrics_total["metrics/precision(B)"] * metrics_total["metrics/recall(B)"]) / (metrics_total["metrics/precision(B)"] + metrics_total["metrics/recall(B)"])
            map50_total = metrics_total["metrics/mAP50(B)"]

            # index 2 map50, index 2 map50-95, index 0 precision and index 1 recall
            f1_person = 2 * (results_person[0] * results_person[1]) / (results_person[0] + results_person[1])
            f1_ball = 2 * (results_ball[0] * results_ball[1]) / (results_ball[0] + results_ball[1])
            map50_person = results_person[2]
            map50_ball = results_ball[2]

            # create wheigted f1 score
            f1_total_weighted = (f1_person + f1_ball) / 2
            map50_total_weighted = (map50_person + map50_ball) / 2

            # log to wandb
            wandb.log({"f1_total": f1_total, "f1_total_weighted": f1_total_weighted,"map50_total_weighted":map50_total_weighted , "map50_total": map50_total, "f1_person": f1_person, "f1_ball": f1_ball, "map50_person": map50_person, "map50_ball": map50_ball})


        except Exception as e:
            print(f"An error occurred while training model {model_name}: {e}")
            return None
        
    wandb.finish()

if __name__ == "__main__":
    # define the sweep configuration
    sweep_config = {
        "name": "yolo_training_sweep",
        "method": "bayes",
        "metric": {
            "name": "f1_total",
            "goal": "maximize"
        },
        "parameters": {
            "model_name": {"value": "PreTrained_Ball_V11.pt"},
            "data_path": {"value": "datasets/YOLOPlayerBall/annotations.yaml"},
            "epochs": {"value": 100},
            "img_size": {"value": 640},
            "devices": {"value": 0},
            "batch_size": {"value": -1},
            "seed": {"value": 42},
            "box": {"distribution": "uniform", "min": 5, "max": 10},
            "cls": {"distribution": "uniform", "min": 0.1, "max": 1},
            "dfl": {"distribution": "uniform", "min": 2, "max": 5}
        }
    }

    # initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project="player-ball-tracking")

    # run the sweep
    wandb.agent(sweep_id, function=train_yolo_model, count=16)