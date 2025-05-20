from Models import (
    # Ready to go models
    StackedAE,
    Combined_CNN,
    # Encoder
    CNN_Spec,
    CNN_Temp,
    TCN_Encoder,
    # Decoder
    CNN_Parallel,
    CNN_Series,
    CostumRNN_Decoder,
    GRU_Decoder
)

model_dict = dict(
    # Ready to go models
    StackedAE=StackedAE,
    Combined_CNN=Combined_CNN,

    # Encoders
    CNN_Spec=CNN_Spec,
    CNN_Temp=CNN_Temp,
    TCN_Encoder=TCN_Encoder,

    # Decoders
    CNN_Parallel=CNN_Parallel,
    CNN_Series=CNN_Series,
    CostumRNN_Decoder=CostumRNN_Decoder,
    GRU_Decoder=GRU_Decoder,
)

def get_model_cls(model_name):
    return model_dict[model_name]
