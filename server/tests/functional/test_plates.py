def test_plates(client):
    res = client.post('/plates', data='qwerty')
    assert res.status_code == 200
    assert res.json ==  ['sssssss', 'xxxxxxx']
