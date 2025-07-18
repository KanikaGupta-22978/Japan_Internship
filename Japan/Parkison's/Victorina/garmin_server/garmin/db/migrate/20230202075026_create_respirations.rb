class CreateRespirations < ActiveRecord::Migration[5.2]
  def change
    create_table :respirations do |t|
      t.references :user, foreign_key: true
      t.date :calendar_date
      t.jsonb :data, null: false, default: {}

      t.timestamps
    end
  end
end
