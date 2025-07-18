class AddOmniauthToUsers < ActiveRecord::Migration[5.2]
  def change
    add_column :users, :provider, :string
    add_column :users, :uid, :string
    add_column :users, :user_access_token, :string
    add_column :users, :user_access_token_secret, :string
  end
end
