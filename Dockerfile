FROM tensorflow/tensorflow:2.14.0

RUN pip install --no-cache-dir \
    pandas \
    scikit-learn \
    matplotlib \
    seaborn \
    numpy


# Copy your code and data into the image
COPY notebooks/Neural_network_model.py /app/notebooks/
COPY data/clean_airbnb_data.csv /app/data/

# Set working directory
WORKDIR /app/notebooks

# Run the script when the container starts
CMD ["python", "Neural_network_model.py"]
