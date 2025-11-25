# Proyecto Segmentación de Vertebras


- Repositorio en Github:
    - URL: https://github.com/jsparadacelis/vertebrae
    - Notas: Dentro del repositorio hay tres carpetas:
        - ml: Contiene datos, notebooks y los modelos
        - vertebrae-backend-api: Contiene todo el código de la API REST de la aplicación.
        - vertebrae-fronten: El código de la aplicación WEB.
    - Las dos carpetas del API y de la aplicación WEB tiene su README para instalar la aplicación en local.

- Aplicación WEB: https://vertebrae-frontend.up.railway.app/
- Funcionamiento:
    - Se puede arrastrar o subir imágenes a través del boton en la aplicación. Recomendamos usar alguna radiografía de estos datasets para hacer pruebas:
        - https://www.kaggle.com/datasets/yasserhessein/the-vertebrae-xray-images
        - https://huggingface.co/datasets/UniqueData/spine-x-ray
    - La aplicación permite hacer la inferencia de las radiografías usando modelos YOLO o Mask R-CNN.
    - La aplicación backend permite configurar los valores del umbral de confianza y de solapamiento entre las segmentaciones a través de variables de entorno.
