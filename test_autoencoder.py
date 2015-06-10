import mnist
import autoencoder


def test_fileobject():

    [train_images, _], _, _ = mnist.load()
    mnist.normalize(train_images)

    f = tt.nnet.sigmoid
    auto = Autoencoder((28, 28), 200, rf_shape=(9, 9), hid_func=f)
    auto.auto_sgd(train_images, rate=0.05, noise=1, n_epochs=3)
    # auto.auto_sgd(train_images, rate=0.1, noise=0, n_epochs=3)

    test = train_images[:1000]
    recs = auto.reconstruct(test)
    print rms(test - recs, axis=1).mean()

    plt.figure(101)
    plt.clf()
    show_recons(test, recs)
    # plt.show()

    auto.to_file('auto.npz')

    auto2 = FileObject.from_file('auto.npz')
    recs2 = auto2.reconstruct(test)
    print rms(test - recs2, axis=1).mean()

    plt.figure(102)
    plt.clf()
    show_recons(test, recs)
    plt.show()
