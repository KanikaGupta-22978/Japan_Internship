class CreateDailies < ActiveRecord::Migration[5.2]
  def change
    create_table :dailies do |t|
      t.references :user, foreign_key: true
      t.jsonb :data, null: false, default: {}

      t.timestamps
    end
  end
end
