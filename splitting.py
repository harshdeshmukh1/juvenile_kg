from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(
    data,
    test_size=0.2,
    random_state=42
)

print("Train size:", len(train_data))
print("Test size:", len(test_data))